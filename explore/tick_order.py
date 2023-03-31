class order_record:
    def __init__(self):
        self.type=""
        self.order=list()

    def from_ask(self,s):
        self.type="Ask"
        for i in range(1,6):
            self.order.append((s[f"AskPrice{i}"],s[f"AskVolume{i}"]))
        return self
    
    def from_bid(self,s):
        self.type="Bid"        
        for i in range(1,6):
            self.order.append((-s[f"BidPrice{i}"],s[f"BidVolume{i}"]))
        return self
    
    def from_num(self,s):
        self.order=sorted(s)
        return self
    
    def copy(self):
        o=order_record()
        o.type=self.type
        o.order=copy.deepcopy(self.order)
        return o

    def copy_type(self):
        o=order_record()
        o.type=self.type
        return o

    def neg(self):
        o=self.copy_type()
        order=o.order
        for (p1,v1) in self.order:
            order.append((p1,-v1))
        return o

    def neg_price(self):
        o=self.copy_type()
        order=o.order
        for (p1,v1) in self.order:
            order.append((-p1,v1))
        return o
    
    def sub_bond(self,r):
        new_r=order_record()
        r3=new_r.order

        r1=self.order
        r2=r.order
        m1=r1[-1][0]
        m2=r2[-1][0]
        m3=min(m1,m2)

        i2=0
        l2=len(r2)

        for i1,(p1,v1) in enumerate(self.order):
            v1_add=0
            while True:
                if i2>=l2:
                    break

                p2,v2=r2[i2]
                if p2>m3:
                    i2+=1
                    break

                if p2>p1:
                    break

                if p2<p1:
                    r3.append((p2,-v2))
                    i2+=1
                    continue

                if p2==p1:
                    v1_add=-v2
                    i2+=1
                    break
            if p1>m3:
                break
            r3.append((p1,v1+v1_add))
        return new_r

    def sub_normal(self,r):
        assert isinstance(r,order_record)
        assert (r.type=="")

        new_r=order_record()
        r3=new_r.order

        r1=self.order
        r2=r.order

        i2=0
        l2=len(r2)

        for i1,(p1,v1) in enumerate(self.order):
            v1_add=0
            while True:
                if i2>=l2:
                    break

                p2,v2=r2[i2]

                if p2>p1:
                    break

                if p2<p1:
                    r3.append((p2,-v2))
                    i2+=1
                    continue

                if p2==p1:
                    v1_add=-v2
                    i2+=1
                    break
            r3.append((p1,v1+v1_add))

        while True:
            if i2>=l2:
                break
            p2,v2=r2[i2]
            r3.append((p2,-v2))
            i2+=1

        return new_r

    def __sub__(self,r):
        if r is None or r is math.nan or r is np.nan:
            return r        
        if r.type=="":
            return self.sub_normal(r)
        return self.sub_bond(r)
    
    def __add__(self,r):
        if r is None or r is math.nan or r is np.nan:
            return r
        r=r.neg()
        if r.type=="":
            return self.sub_normal(r)
        return self.sub_bond(r)

    def __repr__(self):
        return self.order.__repr__()
    



## 以 AskPrice1 的价格 buy MarketBuyVolume 的量
## 以 BidPrice1 的价格 sell MarketSellVolume 的量

bid_append=df.apply(lambda x:order_record().from_num([(-x["BidPrice1"],x["MarketSellVolume"]),(-x["AskPrice1"],x["MarketBuyVolume"])]),axis=1)
bid_tick=df.apply(lambda x:order_record().from_bid(x),axis=1)
bid_tick_lag1=bid_tick.shift(1)
bid_tick_dif=bid_tick-bid_tick_lag1
bid_tick_order=bid_tick_dif+bid_append

ask_append=df.apply(lambda x:order_record().from_num([(x["BidPrice1"],x["MarketSellVolume"]),(x["AskPrice1"],x["MarketBuyVolume"])]),axis=1)
ask_tick=df.apply(lambda x:order_record().from_ask(x),axis=1)
ask_tick_lag1=ask_tick.shift(1)
ask_tick_dif=ask_tick-ask_tick_lag1
ask_tick_order=ask_tick_dif+ask_append

ask_tick_order.iloc[1001]
bid_tick_order.iloc[1001]
ask_append.iloc[1001]

ask_tick_order.iloc[1002]
bid_tick_order.iloc[1002]
ask_append.iloc[1002]
for i in range(500,620):
    print(bid_tick_order.iloc[i])
    #print(bid_tick.iloc[i])
    
bid_tick_order.iloc[1]
bid_tick_order.iloc[-1]
