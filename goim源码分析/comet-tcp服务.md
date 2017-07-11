### 一. tcp简介
tcp.go提供了tcp协议的消息推送，websocket.go则提供了websocket协议的消息推送

### 二. tcp流程
#### 2.1 初始化TCP

```
// InitTCP listen all tcp.bind and start accept connections.
func InitTCP(addrs []string, accept int) (err error) {
	var (
		bind     string
		listener *net.TCPListener
		addr     *net.TCPAddr
	)
	for _, bind = range addrs {
	        // 解析地址字符串，获得一个TCPAddr
		if addr, err = net.ResolveTCPAddr("tcp4", bind); err != nil {
			log.Error("net.ResolveTCPAddr(\"tcp4\", \"%s\") error(%v)", bind, err)
			return
		}
		// 监听地址
		if listener, err = net.ListenTCP("tcp4", addr); err != nil {
			log.Error("net.ListenTCP(\"tcp4\", \"%s\") error(%v)", bind, err)
			return
		}
		if Debug {
			log.Debug("start tcp listen: \"%s\"", bind)
		}
		// split N core accept
		// 每个CPU核上一个go协程去处理TCP连接
		for i := 0; i < accept; i++ {
			go acceptTCP(DefaultServer, listener)
		}
	}
	return
}


// Accept accepts connections on the listener and serves requests
// for each incoming connection.  Accept blocks; the caller typically
// invokes it in a go statement.
func acceptTCP(server *Server, lis *net.TCPListener) {
	var (
		conn *net.TCPConn
		err  error
		r    int
	)
	// 在循环中接受连接，并处理
	for {
	
	        // 接受TCP连接
		if conn, err = lis.AcceptTCP(); err != nil {
			// if listener close then return
			log.Error("listener.Accept(\"%s\") error(%v)", lis.Addr().String(), err)
			return
		}
		
		// 对TCP连接设置TCP keepalive
		if err = conn.SetKeepAlive(server.Options.TCPKeepalive); err != nil {
			log.Error("conn.SetKeepAlive() error(%v)", err)
			return
		}
		
		// 设置发送和接收缓冲区大小
		if err = conn.SetReadBuffer(server.Options.TCPRcvbuf); err != nil {
			log.Error("conn.SetReadBuffer() error(%v)", err)
			return
		}
		if err = conn.SetWriteBuffer(server.Options.TCPSndbuf); err != nil {
			log.Error("conn.SetWriteBuffer() error(%v)", err)
			return
		}
		
		// 创建协程处理新连接
		go serveTCP(server, conn, r)
		if r++; r == maxInt {
			r = 0
		}
	}
}
```
1. 对于每一个监听的地址，创建一个listen socket，然后在每个CPU核上创建一个协程等待接受客户端连接。
2. 对于新连接，设置TCP keepalive，并设置输入输出缓冲区大小，然后创建协程处理连接。


#### 2.2 处理连接
对于新到来的客户端连接，所进行的操作无非是接收客户端数据，然后做相应的处理。serveTCP主要完成的就是这部分的工作，serveTCP的关键点：
1. 流程交互：**将接收数据操作和处理操作拆分开来，在serveTCP中新建了goroutine来执行处理操作（调用dispatchTCP函数），serveTCP所在的 goroutine则专门完成接收数据的操作，两个goroutine之间通过go channel通信。**
2. 数据交互：**为了解决接收数据和处理数据速度不匹配问题，利用Proto环形缓冲区(comet/ring.go)来缓存数据**

```
// auth for goim handshake with client, use rsa & aes.
func (server *Server) authTCP(rr *bufio.Reader, wr *bufio.Writer, p *proto.Proto) (key string, rid int32, heartbeat time.Duration, err error) {
	// 从Reader缓冲区中读取数据，填充Proto
	if err = p.ReadTCP(rr); err != nil {
		return
	}
	// 判断操作类型是否为 AUTH
	if p.Operation != define.OP_AUTH {
		log.Warn("auth operation not valid: %d", p.Operation)
		err = ErrOperation
		return
	}
	// 连接 logic server,
	if key, rid, heartbeat, err = server.operator.Connect(p); err != nil {
		return
	}

	// 向客户端发送AUTH回复
	p.Body = nil
	p.Operation = define.OP_AUTH_REPLY
	if err = p.WriteTCP(wr); err != nil {
		return
	}
	// 发送数据
	err = wr.Flush()
	return
}
```
serveTCP( Server.serveTCP)流程：
1. 从Pool中获取空闲的读写缓冲区，为新连接构造一个Channel。
2. 初始化Channel的读写缓冲区。
3. 准备进行握手操作，首先设置握手连接的超时时间，超时后自动关闭连接。
4. 调用serve.authTCP完成注册验证操作，authTCP会返回连接对应的roomId和之后连接的心跳时间。
5. 注册完成后，准备进行后面的业务操作，首先设定连接的心跳时间。
6. ==关键的一步，新建一个goroutine 用来执行数据处理(dispatchTCP)，当前的gorouine则主要进行数据接收==
7. 在循环中不断的从缓冲区中读取数据，构造Proto对象，存放到Proto缓冲区。然后通知数据处理goroutine

```
func serveTCP(server *Server, conn *net.TCPConn, r int) {
        // 选择定时器和读写缓冲区 
	var (
		// timer
		tr = server.round.Timer(r)
		rp = server.round.Reader(r)
		wp = server.round.Writer(r)
		// ip addr
		lAddr = conn.LocalAddr().String()
		rAddr = conn.RemoteAddr().String()
	)
	if Debug {
		log.Debug("start tcp serve \"%s\" with \"%s\"", lAddr, rAddr)
	}
	server.serveTCP(conn, rp, wp, tr)
}

// TODO linger close?
func (server *Server) serveTCP(conn *net.TCPConn, rp, wp *bytes.Pool, tr *itime.Timer) {
	var (
		err   error
		key   string
		white bool
		hb    time.Duration // heartbeat
		p     *proto.Proto
		b     *Bucket
		trd   *itime.TimerData
		
		// step1 : 从Pool中获取空闲的缓冲区，类型为Buffer*
		rb    = rp.Get()
		wb    = wp.Get()
		ch    = NewChannel(server.Options.CliProto, server.Options.SvrProto, define.NoRoom)
		rr    = &ch.Reader
		wr    = &ch.Writer
	)
	
	// step2 : 初始化channel的读写缓冲区，将step1中获取的空闲缓冲区设置为Reader和Writer的读写缓冲区
	ch.Reader.ResetBuffer(conn, rb.Bytes())
	ch.Writer.ResetBuffer(conn, wb.Bytes())
	// handshake
	// step3 : 设置连接的握手超时时间，超时后直接关闭连接
	trd = tr.Add(server.Options.HandshakeTimeout, func() {
		conn.Close()
	})
	
	// must not setadv, only used in auth
	// step4 : 连接logic服务进行授权操作，同时获得分配的key, RoomId，心跳值
	if p, err = ch.CliProto.Set(); err == nil {
		if key, ch.RoomId, hb, err = server.authTCP(rr, wr, p); err == nil {
			b = server.Bucket(key)
			err = b.Put(key, ch)
		}
	}
	if err != nil {
		conn.Close()
		rp.Put(rb)
		wp.Put(wb)
		tr.Del(trd)
		log.Error("key: %s handshake failed error(%v)", key, err)
		return
	}
	
	// step5 : 重置超时时间
	trd.Key = key
	tr.Set(trd, hb)
	white = DefaultWhitelist.Contains(key)
	if white {
		DefaultWhitelist.Log.Printf("key: %s[%d] auth\n", key, ch.RoomId)
	}
	// hanshake ok start dispatch goroutine
	
	// step6 : 注意这里拆分成两个流程，新建一个 goroutine 去处理数据(dispatchTCP)，同时在当前的goroutine中读取连接
	go server.dispatchTCP(key, conn, wr, wp, wb, ch)
	for {
	
	    // 从Channnel的Proto缓冲区中获取一个Proto对象
		if p, err = ch.CliProto.Set(); err != nil {
			break
		}
		if white {
			DefaultWhitelist.Log.Printf("key: %s start read proto\n", key)
		}
		
		// 从客户端连接中读取数据填充Proto对象
		if err = p.ReadTCP(rr); err != nil {
			break
		}
		if white {
			DefaultWhitelist.Log.Printf("key: %s read proto:%v\n", key, p)
		}
		
		// 如果客户端执行的是心跳操作，则直接设置返回数据
		if p.Operation == define.OP_HEARTBEAT {
			tr.Set(trd, hb)
			p.Body = nil
			p.Operation = define.OP_HEARTBEAT_REPLY
			if Debug {
				log.Debug("key: %s receive heartbeat", key)
			}
		} else {
		    // 执行默认的 Operate
			if err = server.operator.Operate(p); err != nil {
				break
			}
		}
		if white {
			DefaultWhitelist.Log.Printf("key: %s process proto:%v\n", key, p)
		}
		
		// 设置缓冲区写下标 +1并通知处理数据的goroutine
		ch.CliProto.SetAdv()
		ch.Signal()
		
		if white {
			DefaultWhitelist.Log.Printf("key: %s signal\n", key)
		}
	}
	if white {
		DefaultWhitelist.Log.Printf("key: %s server tcp error(%v)\n", key, err)
	}
	if err != nil && err != io.EOF {
		log.Error("key: %s server tcp failed error(%v)", key, err)
	}
	b.Del(key)
	tr.Del(trd)
	rp.Put(rb)
	conn.Close()
	ch.Close()
	if err = server.operator.Disconnect(key, ch.RoomId); err != nil {
		log.Error("key: %s operator do disconnect error(%v)", key, err)
	}
	if white {
		DefaultWhitelist.Log.Printf("key: %s disconnect error(%v)\n", key, err)
	}
	if Debug {
		log.Debug("key: %s server tcp goroutine exit", key)
	}
	return
}


```

dispatchTCP流程：
1. 等待serveTCP通知(serveTCP会执行ch.Signal()通知一个协议数据包读取完毕 )。
2. 如果是结束通知(ProtoFinish)，则设置finish标志，然后跳转到failed。
3. 如果是读取完成通知(ProtoReady)，从缓存中读取serveTCP中构造完成的数据包，然后发送给客户端。
4. 其他情况，直接发送给客户端。
5. 2,3,4步中如果发送失败，则直接跳转到failed。failed中关闭连接然后返回。

```
// dispatch accepts connections on the listener and serves requests
// for each incoming connection.  dispatch blocks; the caller typically
// invokes it in a go statement.
func (server *Server) dispatchTCP(key string, conn *net.TCPConn, wr *bufio.Writer, wp *bytes.Pool, wb *bytes.Buffer, ch *Channel) {
	var (
		err    error
		finish bool
		white  = DefaultWhitelist.Contains(key)
	)
	if Debug {
		log.Debug("key: %s start dispatch tcp goroutine", key)
	}
	for {
		if white {
			DefaultWhitelist.Log.Printf("key: %s wait proto ready\n", key)
		}
		// step1 : 等待通知 (serveTCP会执行ch.Signal()通知 )
		var p = ch.Ready()
		if white {
			DefaultWhitelist.Log.Printf("key: %s proto ready\n", key)
		}
		if Debug {
			log.Debug("key:%s dispatch msg:%v", key, *p)
		}
		switch p {
		// 结束通知
		case proto.ProtoFinish:
			if white {
				DefaultWhitelist.Log.Printf("key: %s receive proto finish\n", key)
			}
			if Debug {
				log.Debug("key: %s wakeup exit dispatch goroutine", key)
			}
			finish = true
			goto failed
		case proto.ProtoReady:
			// fetch message from svrbox(client send)
			// 读取完成通知，从缓存中获取消息，然后写入TCP并从缓存中移除
			for {
				if p, err = ch.CliProto.Get(); err != nil {
					err = nil // must be empty error
					break
				}
				if white {
					DefaultWhitelist.Log.Printf("key: %s start write client proto%v\n", key, p)
				}
				if err = p.WriteTCP(wr); err != nil {
					goto failed
				}
				if white {
					DefaultWhitelist.Log.Printf("key: %s write client proto%v\n", key, p)
				}
				p.Body = nil // avoid memory leak
				ch.CliProto.GetAdv()
			}
		default:
		// 其他的情况，直接发送
			if white {
				DefaultWhitelist.Log.Printf("key: %s start write server proto%v\n", key, p)
			}
			// server send
			if err = p.WriteTCP(wr); err != nil {
				goto failed
			}
			if white {
				DefaultWhitelist.Log.Printf("key: %s write server proto%v\n", key, p)
			}
		}
		if white {
			DefaultWhitelist.Log.Printf("key: %s start flush \n", key)
		}
		// only hungry flush response
		if err = wr.Flush(); err != nil {
			break
		}
		if white {
			DefaultWhitelist.Log.Printf("key: %s flush\n", key)
		}
	}
failed:
	if white {
		DefaultWhitelist.Log.Printf("key: dispatch tcp error(%v)\n", key, err)
	}
	if err != nil {
		log.Error("key: %s dispatch tcp error(%v)", key, err)
	}
	// 返回之前关闭连接
	conn.Close()
	wp.Put(wb)
	// must ensure all channel message discard, for reader won't blocking Signal
	for !finish {
		finish = (ch.Ready() == proto.ProtoFinish)
	}
	if Debug {
		log.Debug("key: %s dispatch goroutine exit", key)
	}
	return
}
```
