### 一. 数据结构
Bucket包含两类关键数据结构：
1. 每个Bucket下包含多个Channel
2. 每个Bucket下包含多个Room( **注意: 每个Room可以包含多个Channel**)
 
Channel既可以属于Room，也可以直接包含在Bucket下面，
Bucket的Channel map中包含了所有属于这个Bucket的Channel (**不管是否属于某个Room**)

```
type BucketOptions struct {
	ChannelSize   int
	RoomSize      int
	RoutineAmount int64
	RoutineSize   int
}

// Bucket is a channel holder.
type Bucket struct {
    // 包含的所有Channel
	cLock    sync.RWMutex        // protect the channels for chs
	chs      map[string]*Channel // map sub key to a channel
	boptions BucketOptions
	
	// room
	// 包含的所有room
	rooms       map[int32]*Room // bucket room channels
	routines    []chan *proto.BoardcastRoomArg
	routinesNum uint64
}
```

### 二. 创建Bucket
在创建Bucket的时候会创建多个Room，并且会创建与Room数量相同的goroutine来执行roomproc，它在循环中处理外界给Room发送的消息。**注意：这里的roomproc和Room并不是一一对应的，创建多个roomproc是为了同时处理多个发送请求，类似于线程池的概念** 
```
// NewBucket new a bucket struct. store the key with im channel.
func NewBucket(boptions BucketOptions) (b *Bucket) {
	b = new(Bucket)
	b.chs = make(map[string]*Channel, boptions.ChannelSize)
	b.boptions = boptions

	// room
	// 这里的重点是创建多个goroutine，用来同时给多个room发送消息
	// b.routines 中包含所有用来通信的 chan
	
	b.rooms = make(map[int32]*Room, boptions.RoomSize)
	b.routines = make([]chan *proto.BoardcastRoomArg, boptions.RoutineAmount)
	for i := int64(0); i < boptions.RoutineAmount; i++ {
		c := make(chan *proto.BoardcastRoomArg, boptions.RoutineSize)
		b.routines[i] = c
		go b.roomproc(c)
	}
	return
}

// roomproc
// 循环等待chan中的消息，将消息发送给对应的room
func (b *Bucket) roomproc(c chan *proto.BoardcastRoomArg) {
	for {
		var (
			arg  *proto.BoardcastRoomArg
			room *Room
		)
		arg = <-c
		if room = b.Room(arg.RoomId); room != nil {
			room.Push(&arg.P)
		}
	}
}

```

### 三. 新增Channel
Channel首先会直接放到Bucket下面，如果配置了Channel属于某个Room，则还要放到对应的Room中。
```
// Put put a channel according with sub key.
func (b *Bucket) Put(key string, ch *Channel) (err error) {
	var (
		room *Room
		ok   bool
	)
	b.cLock.Lock()
	
	// step1 : 将Channel增加到Bucket下面
	b.chs[key] = ch
	
	// step2 : 如果设置Channel属于某个Room，则将Channel添加到对应的Room
	// 如果Room不存在则创建
	if ch.RoomId != define.NoRoom {
		if room, ok = b.rooms[ch.RoomId]; !ok {
			room = NewRoom(ch.RoomId)
			b.rooms[ch.RoomId] = room
		}
	}
	b.cLock.Unlock()
	// 如果指定了RoomId，则加入对应的Room
	if room != nil {
		err = room.Put(ch)
	}
	return
}
```

### 四. 移除Channel
由于Channel可能属于某个Room，因此在移除Channel的时候会从对应的Room中移除，如果移除后Room变为空，则关闭Room
```
// Del delete the channel by sub key.
func (b *Bucket) Del(key string) {
	var (
		ok   bool
		ch   *Channel
		room *Room
	)
	b.cLock.Lock()
	// step1 : 从Bucket的Channel map中移除对应的Channel
	if ch, ok = b.chs[key]; ok {
		delete(b.chs, key)
		if ch.RoomId != define.NoRoom {
			room, _ = b.rooms[ch.RoomId]
		}
	}
	b.cLock.Unlock()
	// step2 : 如果Channel属于Room，则从Room中也移除，如果Room变空，则删除Room
	if room != nil && room.Del(ch) {
		// if empty room, must delete from bucket
		b.DelRoom(ch.RoomId)
	}
}

// DelRoom delete a room by roomid.
func (b *Bucket) DelRoom(rid int32) {
	var room *Room
	b.cLock.Lock()
	if room, _ = b.rooms[rid]; room != nil {
		delete(b.rooms, rid)
	}
	b.cLock.Unlock()
	if room != nil {
		room.Close()
	}
	return
}

```

### 五. 发送消息
1. 向Bucket下面所有的Channel推送消息 (Broadcast)
2. 向某一个Room推送消息 (BroadcastRoom)，采用轮询的方式从routines中选择一个执行发送任务。

```
// 向所有的Channel推送消息
// Broadcast push msgs to all channels in the bucket.
func (b *Bucket) Broadcast(p *proto.Proto) {
	var ch *Channel
	b.cLock.RLock()
	for _, ch = range b.chs {
		// ignore error
		ch.Push(p)
	}
	b.cLock.RUnlock()
}

// 向某个Room推送消息
// BroadcastRoom broadcast a message to specified room
func (b *Bucket) BroadcastRoom(arg *proto.BoardcastRoomArg) {
    
    // 轮询的方式选择roomproc推送消息
	num := atomic.AddUint64(&b.routinesNum, 1) % uint64(b.boptions.RoutineAmount)
	b.routines[num] <- arg
}

// roomproc
func (b *Bucket) roomproc(c chan *proto.BoardcastRoomArg) {
	for {
		var (
			arg  *proto.BoardcastRoomArg
			room *Room
		)
		// 读取消息，获取对应的RoomId，然后发送
		arg = <-c
		if room = b.Room(arg.RoomId); room != nil {
			room.Push(&arg.P)
		}
	}
}

```
### 六. 其他函数

1. 获取Online大于0的所有room id (Online代表room中Channel的数量)，这里利用empty struct作为map的值。

```
// Rooms get all room id where online number > 0.
func (b *Bucket) Rooms() (res map[int32]struct{}) {
	var (
		roomId int32
		room   *Room
	)
	res = make(map[int32]struct{})
	b.cLock.RLock()
	for roomId, room = range b.rooms {
		if room.Online > 0 {
			res[roomId] = struct{}{}
		}
	}
	b.cLock.RUnlock()
	return
}
```

