### 一. 协议格式
comet中客户端和服务端的交互全部通过Proto格式的协议数据包，根据其中的Operation来判断需要执行的操作。
#### 1.1 协议内容
```
// Proto is a request&response written before every goim connect.  It is used internally
// but documented here as an aid to debugging, such as when analyzing
// network traffic.
// tcp:
// binary codec
// websocket & http:
// raw codec, with http header stored ver, operation, seqid
type Proto struct {
	Ver       int16           `json:"ver"`  // protocol version
	Operation int32           `json:"op"`   // operation for request
	SeqId     int32           `json:"seq"`  // sequence number chosen by client
	Body      json.RawMessage `json:"body"` // binary body bytes(json.RawMessage is []byte)
}

```
协议内容分为四个部分：
1. Ver ： 版本号，2字节
2. Operation : 操作，4字节
3. SeqId : 序列号，4字节
4. Body : 内容，不定长字节数组

#### 1.2 协议格式

```
PackSize    : 4字节, 整个协议数据的大小(包含协议头和协议体)
HeaderSize  : 2字节，协议头的大小
Ver         : 2字节，版本号
Operation   : 4字节，操作
SeqId       : 4字节，序列号
Body        : 不定长内容，协议体
```


### 二. 对协议的操作
#### 2.1 构造协议结构体并写入
注意 ***bytes.Writer和bufio.Writer的不同***

bytes.Writer ： 字节缓冲区，不包含其他的内容  
bufio.Writer ： 封装了了io通道和缓冲区
```

func (p *Proto) WriteBodyTo(b *bytes.Writer) (err error) {
	var (
		ph  Proto
		js  []json.RawMessage
		j   json.RawMessage
		jb  []byte
		bts []byte
	)
	offset := int32(PackOffset)
	buf := p.Body[:]
	
	// 循环读取buf中的内容，序列化为json格式
	for {
	        // 如果内容长度小于协议头的大小，则直接break 
		if (len(buf[offset:])) < RawHeaderSize {
			// should not be here
			break
		}
		// step1 : 从buf中读出协议各部分数据，构造Proto结构
		packLen := binary.BigEndian.Int32(buf[offset : offset+HeaderOffset])
		packBuf := buf[offset : offset+packLen]
		// packet
		ph.Ver = binary.BigEndian.Int16(packBuf[VerOffset:OperationOffset])
		ph.Operation = binary.BigEndian.Int32(packBuf[OperationOffset:SeqIdOffset])
		ph.SeqId = binary.BigEndian.Int32(packBuf[SeqIdOffset:RawHeaderSize])
		ph.Body = packBuf[RawHeaderSize:]
		
		// step2 : 将Proto结构序列化为json字符串
		if jb, err = json.Marshal(&ph); err != nil {
			return
		}
		j = json.RawMessage(jb)
		
		// step3 ： 追加到一个总的字符串
		js = append(js, j)
		offset += packLen
	}
	
	// 再次序列化 ？？
	if bts, err = json.Marshal(js); err != nil {
		return
	}
	b.Write(bts)
	return
}


// 向Writer的缓冲区中写入协议数据，包括协议头和协议体
func (p *Proto) WriteTo(b *bytes.Writer) {
	var (
		packLen = RawHeaderSize + int32(len(p.Body))
		buf     = b.Peek(RawHeaderSize)
	)
	
	binary.BigEndian.PutInt32(buf[PackOffset:], packLen)
	binary.BigEndian.PutInt16(buf[HeaderOffset:], int16(RawHeaderSize))
	binary.BigEndian.PutInt16(buf[VerOffset:], p.Ver)
	binary.BigEndian.PutInt32(buf[OperationOffset:], p.Operation)
	binary.BigEndian.PutInt32(buf[SeqIdOffset:], p.SeqId)
	
	if p.Body != nil {
		b.Write(p.Body)
	}
}


func (p *Proto) WriteTCP(wr *bufio.Writer) (err error) {
	var (
		buf     []byte
		packLen int32
	)
	if p.Operation == define.OP_RAW {
		// write without buffer, job concact proto into raw buffer
		// 原始数据，直接写入
		_, err = wr.WriteRaw(p.Body)
		return
	}
	
	// 从wr中获取packLen字节的空间, 先写入头部，再写入消息体
	packLen = RawHeaderSize + int32(len(p.Body))
	if buf, err = wr.Peek(RawHeaderSize); err != nil {
		return
	}
	binary.BigEndian.PutInt32(buf[PackOffset:], packLen)
	binary.BigEndian.PutInt16(buf[HeaderOffset:], int16(RawHeaderSize))
	binary.BigEndian.PutInt16(buf[VerOffset:], p.Ver)
	binary.BigEndian.PutInt32(buf[OperationOffset:], p.Operation)
	binary.BigEndian.PutInt32(buf[SeqIdOffset:], p.SeqId)
	if p.Body != nil {
		_, err = wr.Write(p.Body)
	}
	return
}

// 写入Websocket
func (p *Proto) WriteWebsocket(wr *websocket.Conn) (err error) {
	if p.Body == nil {
		p.Body = emptyJSONBody
	}
	// [{"ver":1,"op":8,"seq":1,"body":{}}, {"ver":1,"op":3,"seq":2,"body":{}}]
	if p.Operation == define.OP_RAW {
		// batch mod
		var b = bytes.NewWriterSize(len(p.Body) + 40*RawHeaderSize)
		if err = p.WriteBodyTo(b); err != nil {
			return
		}
		err = wr.WriteMessage(websocket.TextMessage, b.Buffer())
		return
	}
	// 
	err = wr.WriteJSON([]*Proto{p})
	return
}

```
注意： wr.WriteJSON([]*Proto{p})  
利用 p 构造一个列表，列表中的元素为 *Proto 类型

#### 2.2 读取数据构造协议结构体

利用读取的数据构造Proto对象
```
func (p *Proto) ReadTCP(rr *bufio.Reader) (err error) {
	var (
		bodyLen   int
		headerLen int16
		packLen   int32
		buf       []byte
	)
	if buf, err = rr.Pop(RawHeaderSize); err != nil {
		return
	}
	packLen = binary.BigEndian.Int32(buf[PackOffset:HeaderOffset])
	headerLen = binary.BigEndian.Int16(buf[HeaderOffset:VerOffset])
	p.Ver = binary.BigEndian.Int16(buf[VerOffset:OperationOffset])
	p.Operation = binary.BigEndian.Int32(buf[OperationOffset:SeqIdOffset])
	p.SeqId = binary.BigEndian.Int32(buf[SeqIdOffset:])
	if packLen > MaxPackSize {
		return ErrProtoPackLen
	}
	if headerLen != RawHeaderSize {
		return ErrProtoHeaderLen
	}
	if bodyLen = int(packLen - int32(headerLen)); bodyLen > 0 {
		p.Body, err = rr.Pop(bodyLen)
	} else {
		p.Body = nil
	}
	return
}

func (p *Proto) ReadWebsocket(wr *websocket.Conn) (err error) {
	err = wr.ReadJSON(p)
	return
}


```

