def cal_added_limit_order_volume(snapshot, direction, level):
    """计算限价单挂单量的变化"""
    cur_price = snapshot['%sPrice%d' % (direction, level)]
    cur_volume = snapshot['%sVolume%d' % (direction, level)]

    if direction == 'Bid' and cur_price > snapshot['BidPrice1_Lag1']:
        # 新价格
        return cur_volume
    elif direction == 'Ask' and cur_price < snapshot['AskPrice1_Lag1']:
        # 新价格
        return cur_volume
    elif direction == 'Bid' and cur_price < snapshot['BidPrice5_Lag1']:
        # 原来五档外的买单，假设其很不活跃
        return 0
    elif direction == 'Ask' and cur_price > snapshot['AskPrice5_Lag1']:
        # 原来五档外的买单，假设其很不活跃
        return 0
    else:
        # 假设市价单只发生在原来的买一卖一上
        added_limit_order_volume = 0
        if direction == 'Bid' and cur_price == snapshot['BidPrice1_Lag1']:
            added_limit_order_volume += snapshot['MarketSellVolume']
        elif direction == 'Ask' and cur_price == snapshot['AskPrice1_Lag1']:
            added_limit_order_volume += snapshot['MarketBuyVolume']

        # 查找上一个tick该价位的挂单量，找不到的说明是新价格，上一个tick时挂单量为0
        last_volume = 0
        for i in range(1, 6):
            if cur_price == snapshot['%sPrice%d_Lag1' % (direction, i)]:
                last_volume = snapshot['%sVolume%d_Lag1' % (direction, i)]
                break
        added_limit_order_volume += (cur_volume - last_volume)
        return added_limit_order_volume