## 一. Client 和 Clients
Client和Clients用来管理comet RPC连接，Client封装了rpc.Client和附属数据，Clients是由Client构成的连接池。
```
// Client is rpc client.
type Client struct {
	*rpc.Client
	options ClientOptions
	quit    chan struct{}
	err     error
}

type Clients struct {
	clients []*Client
}

```

## 二. Client
Client的特点在于为了实时的检测连接的可用性，会在循环中每隔一段时间就ping对方，如果ping(或连接)失败，会重新进行连接，在最后分析的Clients的Ping函数中会发现是在goroutine中执行循环的
```
// Rpc client options.
type ClientOptions struct {
	Proto string
	Addr  string
}

// Client is rpc client.
type Client struct {
	*rpc.Client
	options ClientOptions
	quit    chan struct{}
	err     error
}

```
### 2.1 新建Client
```
// Dial connects to an RPC server at the specified network address.
func Dial(options ClientOptions) (c *Client) {
	c = new(Client)
	c.options = options
	c.dial()
	return
}

// Dial connects to an RPC server at the specified network address.
func (c *Client) dial() (err error) {
	var conn net.Conn
	// 建立新的rpc连接
	conn, err = net.DialTimeout(c.options.Proto, c.options.Addr, dialTimeout)
	if err != nil {
		log.Error("net.Dial(%s, %s), error(%v)", c.options.Proto, c.options.Addr, err)
	} else {
		c.Client = rpc.NewClient(conn)
	}
	return
}
```
### 2.2 调用远程服务
Go官方的RPC包中，客户端调用服务的方法有两个：Call和Go。Go是异步调用，Call是同步调用，但官方包中Call和Go都没有提供超时控制，因此goim这里为Client手动实现了Call方法，加上了超时控制
```
// Call invokes the named function, waits for it to complete, and returns its error status.

func (c *Client) Call(serviceMethod string, args interface{}, reply interface{}) (err error) {
	if c.Client == nil {
		err = ErrRpc
		return
	}
	select {
	case call := <-c.Client.Go(serviceMethod, args, reply, make(chan *rpc.Call, 1)).Done:
		err = call.Error
	case <-time.After(callTimeout):
		err = ErrRpcTimeout
	}
	return
}
```
### 2.3 连接保持
为了实时保持对连接可用性的探测，Ping在循环中每隔一段时间发出一次Ping调用，探测服务的可用性，在外界手动调用Close时，Ping循环收到quit channel的退出消息，直接退出返回
```
// Return client error.
func (c *Client) Error() error {
	return c.err
}

// Close client connection.
func (c *Client) Close() {
	c.quit <- struct{}{}
}

// ping ping the rpc connect and reconnect when has an error.
func (c *Client) Ping(serviceMethod string) {
	var (
		arg   = proto.NoArg{}
		reply = proto.NoReply{}
		err   error
	)
	for {
		select {
		// 如果收到退出通知，则返回
		case <-c.quit:
			goto closed
			return
		default:
		}
		
		// 如果连接上未出错，则ping一下，探测远程服务
		if c.Client != nil && c.err == nil {
			// ping
			if err = c.Call(serviceMethod, &arg, &reply); err != nil {
				c.err = err
				if err != rpc.ErrShutdown {
					c.Client.Close()
				}
				log.Error("client.Call(%s, arg, reply) error(%v)", serviceMethod, err)
			}
		} else {
		    // 如果之前出错了，则重新连接远程服务
			// reconnect
			if err = c.dial(); err == nil {
				// reconnect ok
				c.err = nil
				log.Info("client reconnect %s ok", c.options.Addr)
			}
		}
		
		// 每一次ping之间的时间间隔
		time.Sleep(pingDuration)
	}
closed:
	if c.Client != nil {
		c.Client.Close()
	}
}
```

## 三. Clients连接池
Clients主要任务是维持连接池中的连接，对外提供可用的连接

### 3.1 创建连接池
注意：这里并没有检查Dial返回的连接是否可用，只是根据RPC服务地址数创建对应数量的连接
```
// Rpc client options.
type ClientOptions struct {
	Proto string
	Addr  string
}

type Clients struct {
	clients []*Client
}

// Dials connects to RPC servers at the specified network address.
func Dials(options []ClientOptions) *Clients {
	clients := new(Clients)
	for _, op := range options {
		clients.clients = append(clients.clients, Dial(op))
	}
	return clients
}
```
### 3.2 判断和获取可用连接
get函数可以获取一个可用的连接，在上面分析Client的时候，可以看到Client有一个Ping函数，在该函数中不断的发送Ping服务检测远程RPC服务的可用性。这里的可用连接就是根据Ping的结果来判断的。
```
// get get a available client.
func (c *Clients) get() (*Client, error) {
	for _, cli := range c.clients {
		if cli != nil && cli.Client != nil && cli.Error() == nil {
			return cli, nil
		}
	}
	return nil, ErrNoClient
}

// has a available client.
func (c *Clients) Available() (err error) {
	_, err = c.get()
	return
}
```
### 3.3 调用远程服务
Call函数从连接池中选取一个可用的连接，调用RPC服务。Ping函数则对每一个连接启动有一个goroutine，来执行Client的Ping循环，维持心跳检测。
```
// Call invokes the named function, waits for it to complete, and returns its error status.
// this include rpc.Client.Call method, and takes a timeout.
func (c *Clients) Call(serviceMethod string, args interface{}, reply interface{}) (err error) {
	var cli *Client
	if cli, err = c.get(); err == nil {
		err = cli.Call(serviceMethod, args, reply)
	}
	return
}

// Ping the rpc connect and reconnect when has an error.
func (c *Clients) Ping(serviceMethod string) {
	for _, cli := range c.clients {
		go cli.Ping(serviceMethod)
	}
}
```
