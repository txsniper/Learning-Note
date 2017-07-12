## Channel和Room

Channel类似于通道，在逻辑上，将客户端和服务端之间的一条连接封装起来
```
// Channel used by message pusher send msg to write goroutine.
type Channel struct {
	RoomId   int32
	CliProto Ring
	signal   chan *proto.Proto
	Writer   bufio.Writer
	Reader   bufio.Reader
	Next     *Channel
	Prev     *Channel
}
```
1. CliProto : 环形缓冲区，客户端和服务端的数据都会格式化为Proto。
2. signal : chan消息通知，同步客户端和服务端之间的操作，在tcp.go中可以看到用法。
3. Writer : 服务端
4. Reader : 客户端


Room则提供了一个逻辑概念，包含多个连接(即多个Channel)，**向一个Room发送消息实际上也就是向Room中的所有Channel发送消息。**

```
type Room struct {
	id     int32
	rLock  sync.RWMutex
	next   *Channel
	drop   bool
	Online int // dirty read is ok
}
```
## Channel详解
channel.go中代码很少，只提供了几个基本的函数，有很多操作都是外界直接访问Chanenl的数据成员完成(见 tcp.go )  
通过下面的Push函数和Ready/Signal/Close函数可以看出，发往Channel的数据和对Channel的操作都统一化为一个Proto对象，发送给signal。

```
func NewChannel(cli, svr int, rid int32) *Channel {
	c := new(Channel)
	c.RoomId = rid
	c.CliProto.Init(cli)
	c.signal = make(chan *proto.Proto, svr)
	return c
}

// Push server push message.
func (c *Channel) Push(p *proto.Proto) (err error) {
	select {
	case c.signal <- p:
	default:
	}
	return
}

// Ready check the channel ready or close?
func (c *Channel) Ready() *proto.Proto {
	return <-c.signal
}

// Signal send signal to the channel, protocol ready.
func (c *Channel) Signal() {
	c.signal <- proto.ProtoReady
}

// Close close the channel.
func (c *Channel) Close() {
	c.signal <- proto.ProtoFinish
}
```
## Room详解
对Room的各项操作很简单，Room包含的所有Channel都通过一个单链表链接起来，Room结构中提供了两个标识(drop和Online)，Online对Room中的Channel进行计数，drop在Online为0时表示Room被废弃了。**实际上，外界在删除Channel时如果发现Room中不在包含任何Channel，则会删除Room。**

```
// NewRoom new a room struct, store channel room info.
func NewRoom(id int32) (r *Room) {
	r = new(Room)
	r.id = id
	r.drop = false
	r.next = nil
	r.Online = 0
	return
}

// Put put channel into the room.
func (r *Room) Put(ch *Channel) (err error) {
	r.rLock.Lock()
	if !r.drop {
		if r.next != nil {
			r.next.Prev = ch
		}
		ch.Next = r.next
		ch.Prev = nil
		r.next = ch // insert to header
		r.Online++
	} else {
		err = ErrRoomDroped
	}
	r.rLock.Unlock()
	return
}

// Del delete channel from the room.
func (r *Room) Del(ch *Channel) bool {
	r.rLock.Lock()
	if ch.Next != nil {
		// if not footer
		ch.Next.Prev = ch.Prev
	}
	if ch.Prev != nil {
		// if not header
		ch.Prev.Next = ch.Next
	} else {
		r.next = ch.Next
	}
	r.Online--
	r.drop = (r.Online == 0)
	r.rLock.Unlock()
	return r.drop
}

// Push push msg to the room, if chan full discard it.
func (r *Room) Push(p *proto.Proto) {
	r.rLock.RLock()
	for ch := r.next; ch != nil; ch = ch.Next {
		ch.Push(p)
	}
	r.rLock.RUnlock()
	return
}

// Close close the room.
func (r *Room) Close() {
	r.rLock.RLock()
	for ch := r.next; ch != nil; ch = ch.Next {
		ch.Close()
	}
	r.rLock.RUnlock()
}
```