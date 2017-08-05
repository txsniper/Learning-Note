### logic主流程
logic 是一个无状态模块，从功能上看更像一个Proxy，它并不保存任何业务逻辑数据，只是提供了一个中间层，提供各种接口供前后各端访问。logic主要提供的功能如下：
1. 为前面的Comet模块提供了访问Router节点的接口(InitRouter);
2. 提供了向用户Push消息的接口(InitHTTP);


### 提供 Comet 访问 Router 接口
logic主流程通过InitRouter提供了访问Router节点的接口
```
    routerService               = "RouterRPC"
    routerServicePing           = "RouterRPC.Ping"
    routerServicePut            = "RouterRPC.Put"
    routerServiceDel            = "RouterRPC.Del"
    routerServiceDelServer      = "RouterRPC.DelServer"
    routerServiceAllRoomCount   = "RouterRPC.AllRoomCount"
    routerServiceAllServerCount = "RouterRPC.AllServerCount"
    routerServiceGet            = "RouterRPC.Get"
    routerServiceMGet           = "RouterRPC.MGet"
    routerServiceGetAll         = "RouterRPC.GetAll"
```
接口中"RouterRPC.Put"和"RouterRPC.Del"用于新连接的处理和连接的断开，其它的接口则主要完成业务逻辑。

### 提供向用户Push消息的接口
goim使用kafka向用户push消息，logic模块通过两个步骤提供了push消息的功能：
1. InitHTTP ：对外提供接口，其他模块可以调用这些接口向用户push消息。
```
    httpServeMux.HandleFunc("/1/push", Push)
    httpServeMux.HandleFunc("/1/pushs", Pushs)
    httpServeMux.HandleFunc("/1/push/all", PushAll)
    httpServeMux.HandleFunc("/1/push/room", PushRoom)
    httpServeMux.HandleFunc("/1/server/del", DelServer)
    httpServeMux.HandleFunc("/1/count", Count)
```
2. InitKafka ：提供了访问Kafka的基础接口，InitHTTP在底层会调用它们来Push消息。
```
func mpushKafka(serverId int32, keys []string, msg []byte) (err error) {
	var (
		vBytes []byte
		v      = &proto.KafkaMsg{OP: define.KAFKA_MESSAGE_MULTI, ServerId: serverId, SubKeys: keys, Msg: msg}
	)
	if vBytes, err = json.Marshal(v); err != nil {
		return
	}
	producer.Input() <- &sarama.ProducerMessage{Topic: KafkaPushsTopic, Value: sarama.ByteEncoder(vBytes)}
	return
}

func broadcastKafka(msg []byte) (err error) {
	var (
		vBytes []byte
		v      = &proto.KafkaMsg{OP: define.KAFKA_MESSAGE_BROADCAST, Msg: msg}
	)
	if vBytes, err = json.Marshal(v); err != nil {
		return
	}
	producer.Input() <- &sarama.ProducerMessage{Topic: KafkaPushsTopic, Value: sarama.ByteEncoder(vBytes)}
	return
}

func broadcastRoomKafka(rid int32, msg []byte, ensure bool) (err error) {
	var (
		vBytes   []byte
		ridBytes [4]byte
		v        = &proto.KafkaMsg{OP: define.KAFKA_MESSAGE_BROADCAST_ROOM, RoomId: rid, Msg: msg, Ensure: ensure}
	)
	if vBytes, err = json.Marshal(v); err != nil {
		return
	}
	binary.BigEndian.PutInt32(ridBytes[:], rid)
	producer.Input() <- &sarama.ProducerMessage{Topic: KafkaPushsTopic, Key: sarama.ByteEncoder(ridBytes[:]), Value: sarama.ByteEncoder(vBytes)}
	return
}
```


