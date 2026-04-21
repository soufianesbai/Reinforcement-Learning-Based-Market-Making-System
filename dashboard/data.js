window.MarketMindData = {
  products: ["SOLUSDT", "BTCUSDT", "ETHUSDT", "BNBUSDT"],
  agents: ["RL Agent", "Mean Reversion", "RSI", "Bollinger"],
  orderBook: {
    ask: 150.25,
    bid: 149.85,
    asks: [
      { price: 150.25, quantity: 1.2 },
      { price: 150.3, quantity: 0.9 },
      { price: 150.35, quantity: 1.6 }
    ],
    bids: [
      { price: 149.85, quantity: 1.4 },
      { price: 149.8, quantity: 1.1 },
      { price: 149.75, quantity: 1.8 }
    ]
  },
  candles: [
    { o: 149.8, h: 150.4, l: 149.5, c: 150.2 },
    { o: 150.2, h: 150.8, l: 149.9, c: 150.0 },
    { o: 150.0, h: 151.1, l: 149.8, c: 150.9 },
    { o: 150.9, h: 151.2, l: 150.3, c: 150.5 },
    { o: 150.5, h: 151.0, l: 150.2, c: 150.8 },
    { o: 150.8, h: 152.0, l: 150.6, c: 151.7 },
    { o: 151.7, h: 152.2, l: 151.1, c: 151.3 },
    { o: 151.3, h: 151.9, l: 150.9, c: 151.8 },
    { o: 151.8, h: 152.5, l: 151.5, c: 152.1 },
    { o: 152.1, h: 152.4, l: 151.0, c: 151.4 },
    { o: 151.4, h: 151.8, l: 150.8, c: 151.0 },
    { o: 151.0, h: 151.6, l: 150.6, c: 151.5 }
  ],
  priceAxis: [150.0, 150.5, 151.0, 151.5, 152.0, 152.5],
  performance: {
    ticks: ["10:00", "10:05", "10:10", "10:15", "10:20", "10:25"],
    series: [0, 22, 18, 40, 37, 64, 52, 79, 74, 96, 93, 112]
  },
  tradeLog: [
    { time: "10:03", action: "Buy", result: -25.5 },
    { time: "10:07", action: "Sell", result: 50.0 },
    { time: "10:12", action: "Buy", result: -149.75 }
  ]
};
