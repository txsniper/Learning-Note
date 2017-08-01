### 一致性Hash
goim中使用ketama算法来分配router服务

```
// 虚拟节点
type node struct {
	node string         // 节点名
	hash uint           // 节点hash
}

// 为了后续对tickArray进行排序(Sort)，对tickArray实现了Len,Less,Swap方法，这三个方法会在Sort中调用
type tickArray []node

func (p tickArray) Len() int           { return len(p) }
func (p tickArray) Less(i, j int) bool { return p[i].hash < p[j].hash }
func (p tickArray) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }
func (p tickArray) Sort()              { sort.Sort(p) }

type HashRing struct {
	defaultSpots int            // 每一个实体节点对应的虚拟节点的基数
	ticks        tickArray
	length       int
}

```
### 新增节点
```
// Adds a new node to a hash ring
// n: name of the server
// s: multiplier for default number of ticks (useful when one cache node has more resources, like RAM, than another)
// s: 倍数，用于针对单个节点动态调整虚拟节点数，比如当某个节点上包含其他很多业务时，我们可以利用它将该实体节点对应的虚拟节点减少，减小压力
func (h *HashRing) AddNode(n string, s int) {
	tSpots := h.defaultSpots * s
	hash := sha1.New()
	for i := 1; i <= tSpots; i++ {
		hash.Write([]byte(n + ":" + strconv.Itoa(i)))
		hashBytes := hash.Sum(nil)

		n := &node{
			node: n,
			hash: uint(hashBytes[19]) | uint(hashBytes[18])<<8 | uint(hashBytes[17])<<16 | uint(hashBytes[16])<<24,
		}
        // 将节点加入到列表中
		h.ticks = append(h.ticks, *n)
		hash.Reset()
	}
}

```
### 调整数据
将所有的节点都加入之后，需要对节点列表进行排序，为后续一致性Hash计算做准备
```
func (h *HashRing) Bake() {
	h.ticks.Sort()
	h.length = len(h.ticks)
}
```
### 获取节点
```
func (h *HashRing) Hash(s string) string {
    hash := sha1.New()
    hash.Write([]byte(s))
    hashBytes := hash.Sum(nil)
    v := uint(hashBytes[19]) | uint(hashBytes[18])<<8 | uint(hashBytes[17])<<16 | uint(hashBytes[16])<<24

    // Search uses binary search to find and return the smallest index i in [0, n) at which f(i) is true
    i := sort.Search(h.length, func(i int) bool { return h.ticks[i].hash >= v })

    // 如果 i 到达最后，则分配到第一个节点
    if i == h.length {
    	i = 0
    }

    return h.ticks[i].node
}

```
