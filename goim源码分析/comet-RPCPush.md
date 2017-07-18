## RPCPush服务
comet启动阶段，会执行InitRPCPush函数启动Push接口，用于转发后端Push服务给客户端的Push消息。

## 整体流程
### 启动PRC接口
1. 注册服务，启动监听，对于每一个地址启动一个专门的goroutine处理push任务。
```
func InitRPCPush(addrs []string) (err error) {
	var (
		bind          string
		network, addr string
		c             = &PushRPC{}
	)
	rpc.Register(c)
	for _, bind = range addrs {
		if network, addr, err = inet.ParseNetwork(bind); err != nil {
			log.Error("inet.ParseNetwork() error(%v)", err)
			return
		}
		go rpcListen(network, addr)
	}
	return
}

func rpcListen(network, addr string) {
	l, err := net.Listen(network, addr)
	if err != nil {
		log.Error("net.Listen(\"%s\", \"%s\") error(%v)", network, addr, err)
		panic(err)
	}
	// if process exit, then close the rpc addr
	defer func() {
		log.Info("listen rpc: \"%s\" close", addr)
		if err := l.Close(); err != nil {
			log.Error("listener.Close() error(%v)", err)
		}
	}()
	rpc.Accept(l)
}
```
### Push消息
在PushRPC接口收到后端发送的消息之后，根据各种不同的需求向客户端对应的Channel Push消息，而每个处理客户端连接的goroutine会在dispatchTCP (tcp.go) 中接收信息，然后转发给客户端。
```
// Push RPC
type PushRPC struct {
}

func (this *PushRPC) Ping(arg *proto.NoArg, reply *proto.NoReply) error {
	return nil
}

// Push push a message to a specified sub key
func (this *PushRPC) PushMsg(arg *proto.PushMsgArg, reply *proto.NoReply) (err error) {
	var (
		bucket  *Bucket
		channel *Channel
	)
	if arg == nil {
		err = ErrPushMsgArg
		return
	}
	bucket = DefaultServer.Bucket(arg.Key)
	if channel = bucket.Channel(arg.Key); channel != nil {
		err = channel.Push(&arg.P)
	}
	return
}

// Push push a message to specified sub keys
// 向多个key对应的Channel发送消息
func (this *PushRPC) MPushMsg(arg *proto.MPushMsgArg, reply *proto.MPushMsgReply) (err error) {
	var (
		bucket  *Bucket
		channel *Channel
		key     string
		n       int
	)
	reply.Index = -1
	if arg == nil {
		err = ErrMPushMsgArg
		return
	}
	for n, key = range arg.Keys {
		bucket = DefaultServer.Bucket(key)
		if channel = bucket.Channel(key); channel != nil {
			if err = channel.Push(&arg.P); err != nil {
				return
			}
			reply.Index = int32(n)
		}
	}
	return
}

// MPushMsgs push msgs to multiple user.
func (this *PushRPC) MPushMsgs(arg *proto.MPushMsgsArg, reply *proto.MPushMsgsReply) (err error) {
	var (
		bucket  *Bucket
		channel *Channel
		n       int32
		PMArg   *proto.PushMsgArg
	)
	reply.Index = -1
	if arg == nil {
		err = ErrMPushMsgsArg
		return
	}
	for _, PMArg = range arg.PMArgs {
		bucket = DefaultServer.Bucket(PMArg.Key)
		if channel = bucket.Channel(PMArg.Key); channel != nil {
			if err = channel.Push(&PMArg.P); err != nil {
				return
			}
			n++
			reply.Index = n
		}
	}
	return
}

// Broadcast broadcast msg to all user.
func (this *PushRPC) Broadcast(arg *proto.BoardcastArg, reply *proto.NoReply) (err error) {
	var bucket *Bucket
	for _, bucket = range DefaultServer.Buckets {
		go bucket.Broadcast(&arg.P)
	}
	return
}

// Broadcast broadcast msg to specified room.
func (this *PushRPC) BroadcastRoom(arg *proto.BoardcastRoomArg, reply *proto.NoReply) (err error) {
	var bucket *Bucket
	for _, bucket = range DefaultServer.Buckets {
		bucket.BroadcastRoom(arg)
	}
	return
}

func (this *PushRPC) Rooms(arg *proto.NoArg, reply *proto.RoomsReply) (err error) {
	var (
		roomId  int32
		bucket  *Bucket
		roomIds = make(map[int32]struct{})
	)
	for _, bucket = range DefaultServer.Buckets {
		for roomId, _ = range bucket.Rooms() {
			roomIds[roomId] = struct{}{}
		}
	}
	reply.RoomIds = roomIds
	return
}
```
