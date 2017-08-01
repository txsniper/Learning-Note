## 概述
comet中的Channel封装了一条客户端和服务端的连接，连接中使用的读写缓存和定时器都来自Round对象。
Round对象管理着多个资源池，在分配给Channel资源时，会直接返回一个资源池，对于不同的Channel，使用了不同的资源池。**这样设计的主要作用是降低高并发下多个channel对锁的争用**
下面将结合Round.go , tcp.go来详细分析资源是如何分配和回收的。

## 资源获取
tcp.go在接收新连接时会进行计数，Round对象会根据这个计数来分配资源池。
```
// tcp.go acceptTCP

go serveTCP(server, conn, r)
if r++; r == maxInt {
	r = 0
}

```
在serveTCP中，Round对象根据计数r来获取对应的资源，**注意从round中获取的缓存是Pool类型的，实际上获取的不是一个单一的缓存块儿，而是一个缓存分配池。**

```
func serveTCP(server *Server, conn *net.TCPConn, r int) {
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
```

在server.serveTCP中，会从上面获取的缓存分配池中获取真正使用的缓冲区。
```
func (server *Server) serveTCP(conn *net.TCPConn, rp, wp *bytes.Pool, tr *itime.Timer) {
	var (
		err   error
		key   string
		white bool
		hb    time.Duration // heartbeat
		p     *proto.Proto
		b     *Bucket
		trd   *itime.TimerData

                // 注意这里的Get函数
		rb    = rp.Get()
		wb    = wp.Get()
		ch    = NewChannel(server.Options.CliProto, server.Options.SvrProto, define.NoRoom)
		rr    = &ch.Reader
		wr    = &ch.Writer
	)
	ch.Reader.ResetBuffer(conn, rb.Bytes())
	ch.Writer.ResetBuffer(conn, wb.Bytes())
    .......
}
```
调用缓存分配池的Get函数来获取Channel真正使用的缓存块儿。

```
// libs/bytes/buffer.go
// Get get a free memory buffer.
func (p *Pool) Get() (b *Buffer) {
	p.lock.Lock()
	if b = p.free; b == nil {
		p.grow()
		b = p.free
	}
	p.free = b.next
	p.lock.Unlock()
	return
}

// Put put back a memory buffer to free.
func (p *Pool) Put(b *Buffer) {
	p.lock.Lock()
	b.next = p.free
	p.free = b
	p.lock.Unlock()
	return
}
```
在Get函数中，当空闲的缓冲区不够时会调用grow函数来获取新的缓冲区。

## 资源回收
在serveTCP和dispatchTCP两个函数中，当资源使用完成后，直接将获取的缓冲区归还给资源池，并从定时器管理器中删除对应的定时器。

```
// serveTCP
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

// dispatchTCP
failed:
	if white {
		DefaultWhitelist.Log.Printf("key: dispatch tcp error(%v)\n", key, err)
	}
	if err != nil {
		log.Error("key: %s dispatch tcp error(%v)", key, err)
	}
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
    
```