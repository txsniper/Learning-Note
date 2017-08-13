### Job主流程
job是Kakfa的消费者，主要将Kakfa中的消息发送给comet模块，job可以进行横向扩展以分摊压力。

#### 从Kakfa收消息 (InitKafka)
kafka使用zookeeper来**实现动态的集群扩展**，不需要更改客户端（producer和consumer）的配置。  
broker会在zookeeper注册并保持相关的元数据（topic，partition信息等）更新。而客户端会在zookeeper上注册相关的watcher。一旦zookeeper发生变化，客户端能及时感知并作出相应调整。这样就保证了添加或去除broker时，各broker间仍能自动实现负载均衡。这里的客户端指的是Kafka的消息生产端(Producer)和消息消费端(Consumer)Producer端使用zookeeper用来"发现"broker列表,以及和Topic下每个partition的leader建立socket连接并发送消息。  
也就是说每个Topic的partition是由Lead角色的Broker端使用zookeeper来注册broker信息,以及监测partition leader存活性.Consumer端使用zookeeper用来注册consumer信息,其中包括consumer消费的partition列表等,同时也用来发现broker列表,并和partition leader建立socket连接,并获取消息。

```
    config := consumergroup.NewConfig()
	config.Offsets.Initial = sarama.OffsetNewest
	config.Offsets.ProcessingTimeout = OFFSETS_PROCESSING_TIMEOUT_SECONDS
	config.Offsets.CommitInterval = OFFSETS_COMMIT_INTERVAL
	config.Zookeeper.Chroot = Conf.ZKRoot
	kafkaTopics := []string{Conf.KafkaTopic}
    
    // job 加入 ConsumerGroup 
	cg, err := consumergroup.JoinConsumerGroup(KAFKA_GROUP_NAME, kafkaTopics, Conf.ZKAddrs, config)
	if err != nil {
		return err
	}
	go func() {
		for err := range cg.Errors() {
			log.Error("consumer error(%v)", err)
		}
	}()
	go func() {
		for msg := range cg.Messages() {
			log.Info("deal with topic:%s, partitionId:%d, Offset:%d, Key:%s msg:%s", msg.Topic, msg.Partition, msg.Offset, msg.Key, msg.Value)

            // push消息
			push(msg.Value)

            // 消费消息之后，发送 ack 
			cg.CommitUpto(msg)
		}
	}()
```

#### 消息Push (InitPush)

1. 创建多个 goroutine 来 处理Push任务，使用轮询的方式分配处理任务。
2. push函数中会对每个Push任务进行区分，判断各种广播方式(所有的发送方式都依赖于InitComet中的向Comet发送消息接口, 向单个room广播消息也只是在之上做了一层简单的封装)。
```
type pushArg struct {
	ServerId int32
	SubKeys  []string
	Msg      []byte
	RoomId   int32
}

var (
	pushChs []chan *pushArg
)

func InitPush() {
	pushChs = make([]chan *pushArg, Conf.PushChan)
	for i := 0; i < Conf.PushChan; i++ {
		pushChs[i] = make(chan *pushArg, Conf.PushChanSize)
		go processPush(pushChs[i])
	}
}

func processPush(ch chan *pushArg) {
	var arg *pushArg
	for {
		arg = <-ch
		mPushComet(arg.ServerId, arg.SubKeys, arg.Msg)
	}
}

func push(msg []byte) (err error) {
	m := &proto.KafkaMsg{}
	if err = json.Unmarshal(msg, m); err != nil {
		log.Error("json.Unmarshal(%s) error(%s)", msg, err)
		return
	}
	switch m.OP {

    // Push 给多个 Key 的用户
	case define.KAFKA_MESSAGE_MULTI:
		pushChs[rand.Int()%Conf.PushChan] <- &pushArg{ServerId: m.ServerId, SubKeys: m.SubKeys, Msg: m.Msg, RoomId: define.NoRoom}

    // 全员广播
	case define.KAFKA_MESSAGE_BROADCAST:
		broadcast(m.Msg)

    // 向一个room广播
	case define.KAFKA_MESSAGE_BROADCAST_ROOM:

        // 获取 room id
		room := roomBucket.Get(int32(m.RoomId))
		if m.Ensure {
			go room.EPush(0, define.OP_SEND_SMS_REPLY, m.Msg)
		} else {
			err = room.Push(0, define.OP_SEND_SMS_REPLY, m.Msg)
			if err != nil {
				log.Error("room.Push(%s) roomId:%d error(%v)", m.Msg, err)
			}
		}
	default:
		log.Error("unknown operation:%s", m.OP)
	}
	return
}

```
#### 向Room push消息 (InitRoomBucket)
初始化RoomBucket，管理了所有需要的Room，在初始化时并没有为所有的room创建Room结构，在需要Push消息的时候进行创建(Get函数中创建)
```
type RoomBucket struct {
	roomNum int
	rooms   map[int32]*Room
	rwLock  sync.RWMutex
	options RoomOptions
	round   *Round
}

func InitRoomBucket(r *Round, options RoomOptions) {
	roomBucket = &RoomBucket{
		roomNum: 0,
		rooms:   make(map[int32]*Room, roomMapCup),
		rwLock:  sync.RWMutex{},
		options: options,
		round:   r,
	}
}

func (this *RoomBucket) Get(roomId int32) (r *Room) {
	this.rwLock.Lock()
	room, ok := this.rooms[roomId]
    // 如果未找到Room结构则创建
	if !ok {
		room = NewRoom(roomId, this.round.Timer(this.roomNum), this.options)
		this.rooms[roomId] = room
		this.roomNum++
		log.Debug("new roomId:%d num:%d", roomId, this.roomNum)
	}
	this.rwLock.Unlock()
	return room
}

```
向Room push消息，有两种方式:
1. Push : 不保证一定发送，如果channel已经满了，则会丢弃消息(这里利用select读写channle是非阻塞的特性)
2. EPush : 保证一定发送，如果channel已满，则等待
```
// Push push msg to the room, if chan full discard it.
func (r *Room) Push(ver int16, operation int32, msg []byte) (err error) {
	var p = &proto.Proto{Ver: ver, Operation: operation, Body: msg}
	select {
	case r.proto <- p:
	default:
		err = ErrRoomFull
	}
	return
}

// EPush ensure push msg to the room.
func (r *Room) EPush(ver int16, operation int32, msg []byte) {
	var p = &proto.Proto{Ver: ver, Operation: operation, Body: msg}
	r.proto <- p
	return
}
```

对于每一个room 的Push任务，都会启动一个goroutine来进行处理，采用批量push的方式
```
// NewRoom new a room struct, store channel room info.
// 向一个Room 批量push消息的流程，两个因素：
// 1. 每批的大小，每批最多 BatchNum
// 2. 间隔时间，消息缓存时间最多 1s
func NewRoom(id int32, t *itime.Timer, options RoomOptions) (r *Room) {
	r = new(Room)
	r.id = id
	// 创建一个channel，BatchNum : 批量push的每批数量
	r.proto = make(chan *proto.Proto, options.BatchNum*2)
	go r.pushproc(t, options.BatchNum, options.SignalTime)
	return
}

func (r *Room) pushproc(timer *itime.Timer, batch int, sigTime time.Duration) {
	var (
		n    int
		last time.Time
		p    *proto.Proto
		td   *itime.TimerData
		buf  = bytes.NewWriterSize(int(proto.MaxBodySize))
	)
	log.Debug("start room: %d goroutine", r.id)

	// 设置一个Timer，用于批量发送的时间限制
	td = timer.Add(sigTime, func() {
		select {
		case r.proto <- roomReadyProto:
		default:
		}
	})
	for {
		if p = <-r.proto; p == nil {
			break // exit
		} else if p != roomReadyProto {
			// merge buffer ignore error, always nil
			p.WriteTo(buf)

			// 如果n == 1，说明开始一个批次的push，则设置Timer再次等待数据
			if n++; n == 1 {
				last = time.Now()
				timer.Set(td, sigTime)
				continue
			} else if n < batch {
				// 更新等待时间
				if sigTime > time.Now().Sub(last) {
					continue
				}
			}
		} else {
			// 如果没有缓存的消息，则直接 continue，如果有的话，调用 broadcastRoomBytes
			if n == 0 {
				continue
			}
		}
		broadcastRoomBytes(r.id, buf.Buffer())
		n = 0
		// TODO use reset buffer
		// after push to room channel, renew a buffer, let old buffer gc
		buf = bytes.NewWriterSize(buf.Size())
	}
	timer.Del(td)
	log.Debug("room: %d goroutine exit", r.id)
}
```

#### 向Comet模块Push消息(InitComet)
InitComet流程中提供了向Comet模块Push消息的接口，上面的所有Push任务底层都是调用它们
由于不同的Push任务消息格式都不一样，因此对每一类消息都会创建一个Channel数组用于分摊消息处理任务
```
type CometOptions struct {
	RoutineSize int64
	RoutineChan int
}

type Comet struct {
	serverId  int32
	rpcClient *xrpc.Clients

	// 每个 goroutine 都会使用一个Channel来处理Push消息
	pushRoutines      []chan *proto.MPushMsgArg
	broadcastRoutines []chan *proto.BoardcastArg
	roomRoutines      []chan *proto.BoardcastRoomArg

	// 操作过程中的下标，用于轮询使用 Comet结构中的 goroutine
	pushRoutinesNum      int64
	roomRoutinesNum      int64
	broadcastRoutinesNum int64

	options CometOptions
}

func InitComet(addrs map[int32]string, options CometOptions) (err error) {
	var (
		serverId      int32
		bind          string
		network, addr string
	)
	for serverId, bind = range addrs {
		var rpcOptions []xrpc.ClientOptions
		for _, bind = range strings.Split(bind, ",") {
			if network, addr, err = inet.ParseNetwork(bind); err != nil {
				log.Error("inet.ParseNetwork() error(%v)", err)
				return
			}
			options := xrpc.ClientOptions{
				Proto: network,
				Addr:  addr,
			}
			rpcOptions = append(rpcOptions, options)
		}
		// rpc clients
		rpcClient := xrpc.Dials(rpcOptions)
		// ping & reconnect
		rpcClient.Ping(CometServicePing)
		// comet
		c := new(Comet)
		c.serverId = serverId
		c.rpcClient = rpcClient
		// 创建3个channel : 向多个用户Push消息；向Room 广播；向全部广播
		c.pushRoutines = make([]chan *proto.MPushMsgArg, options.RoutineSize)
		c.roomRoutines = make([]chan *proto.BoardcastRoomArg, options.RoutineSize)
		c.broadcastRoutines = make([]chan *proto.BoardcastArg, options.RoutineSize)
		c.options = options

        // 保存Comet Service信息，每一个service对应一个Comet
		cometServiceMap[serverId] = c
		// process
		// 启动多个goroutine处理push请求，每个gorountine都有自己的Channel
		for i := int64(0); i < options.RoutineSize; i++ {
			pushChan := make(chan *proto.MPushMsgArg, options.RoutineChan)
			roomChan := make(chan *proto.BoardcastRoomArg, options.RoutineChan)
			broadcastChan := make(chan *proto.BoardcastArg, options.RoutineChan)
			c.pushRoutines[i] = pushChan
			c.roomRoutines[i] = roomChan
			c.broadcastRoutines[i] = broadcastChan
			go c.process(pushChan, roomChan, broadcastChan)
		}
		log.Info("init comet rpc: %v", rpcOptions)
	}
	return
}
```

根据消息的类型决定调用的接口
```
func (c *Comet) process(pushChan chan *proto.MPushMsgArg, roomChan chan *proto.BoardcastRoomArg, broadcastChan chan *proto.BoardcastArg) {
	var (
		pushArg      *proto.MPushMsgArg
		roomArg      *proto.BoardcastRoomArg
		broadcastArg *proto.BoardcastArg
		reply        = &proto.NoReply{}
		err          error
	)
	for {
		select {
		case pushArg = <-pushChan:
			// push
			err = c.rpcClient.Call(CometServiceMPushMsg, pushArg, reply)
			if err != nil {
				log.Error("rpcClient.Call(%s, %v, reply) serverId:%d error(%v)", CometServiceMPushMsg, pushArg, c.serverId, err)
			}
			pushArg = nil
		case roomArg = <-roomChan:
			// room
			err = c.rpcClient.Call(CometServiceBroadcastRoom, roomArg, reply)
			if err != nil {
				log.Error("rpcClient.Call(%s, %v, reply) serverId:%d error(%v)", CometServiceBroadcastRoom, roomArg, c.serverId, err)
			}
			roomArg = nil
		case broadcastArg = <-broadcastChan:
			// broadcast
			err = c.rpcClient.Call(CometServiceBroadcast, broadcastArg, reply)
			if err != nil {
				log.Error("rpcClient.Call(%s, %v, reply) serverId:%d error(%v)", CometServiceBroadcast, broadcastArg, c.serverId, err)
			}
			broadcastArg = nil
		}
	}
}
```

#### 动态获取所有Room信息
