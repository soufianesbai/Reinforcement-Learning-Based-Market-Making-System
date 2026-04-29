window.MarketMindComponents = (function () {
  function renderBookRows(rows, side) {
    const safeRows = Array.isArray(rows) ? rows : [];
    if (!safeRows.length) {
      return `<div class="orderbook-empty">Waiting for live data...</div>`;
    }

    return safeRows
      .slice(0, 8)
      .map((row) => `
        <div class="orderbook-row ${side}">
          <span class="price">${Number(row.price).toFixed(2)}</span>
          <span class="qty">${Number(row.quantity).toFixed(4)}</span>
        </div>
      `)
      .join("");
  }

  function topbar() {
    return `
      <header class="topbar">
        <div class="brand">
          <div class="brand-icon" aria-hidden="true">
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
              <path d="M2 14L6 10L9 12L14 6L18 8" stroke="#4aa3ff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
              <path d="M2 17H18" stroke="#9fb3d1" stroke-width="1" opacity=".65"/>
            </svg>
          </div>
          <div class="brand-text">
            <h1>MarketMind</h1>
            <p>Live Testnet Trading</p>
          </div>
        </div>
        <div class="top-actions">
          <div class="topbar-price-badge">
            <span class="topbar-price-label">MID</span>
            <span class="topbar-price-value" id="topbar-price">--</span>
          </div>
        </div>
      </header>
    `;
  }

  function statCards() {
    return `
      <div class="stat-cards">
        <div class="stat-card active" data-panel="inventory">
          <div class="stat-card-icon">📦</div>
          <div class="stat-card-body">
            <span class="stat-card-label" id="inventory-label">Inventory</span>
            <span class="stat-card-value" id="side-inventory">--</span>
          </div>
        </div>
        <div class="stat-card" data-panel="cash">
          <div class="stat-card-icon">💵</div>
          <div class="stat-card-body">
            <span class="stat-card-label">Cash (USDT)</span>
            <span class="stat-card-value" id="side-cash">--</span>
          </div>
        </div>
        <div class="stat-card" data-panel="history">
          <div class="stat-card-icon">🕘</div>
          <div class="stat-card-body">
            <span class="stat-card-label">Total Trades</span>
            <span class="stat-card-value" id="side-history">--</span>
          </div>
        </div>
      </div>
    `;
  }

  function orderBookCard(data) {
    // Robustesse : valeurs par défaut si null/NaN
    const askVal = (data.ask != null && !isNaN(data.ask)) ? data.ask.toFixed(2) : '--';
    const bidVal = (data.bid != null && !isNaN(data.bid)) ? data.bid.toFixed(2) : '--';
    let spreadVal = '--';
    if (data.ask != null && data.bid != null && !isNaN(data.ask) && !isNaN(data.bid)) {
      spreadVal = (data.ask - data.bid).toFixed(2);
    }
    return `
      <section class="card">
        <h2>Live Order Book</h2>
        <div class="orderbook-viz">
          <div class="order-asks"></div>
          <div class="order-bids"></div>
          <div class="midline"></div>
          <div class="badge-stack">
            <span class="badge ask">Ask ${askVal}</span>
            <span class="spread-badge" id="spread-badge">Spread ${spreadVal}</span>
            <span class="badge bid">Bid ${bidVal}</span>
          </div>
        </div>
        <div class="orderbook-depth">
          <div class="orderbook-side">
            <div class="orderbook-side-title ask-text">Top Asks</div>
            <div id="orderbook-asks" class="orderbook-rows">${renderBookRows(data.asks, "ask")}</div>
          </div>
          <div class="orderbook-side">
            <div class="orderbook-side-title bid-text">Top Bids</div>
            <div id="orderbook-bids" class="orderbook-rows">${renderBookRows(data.bids, "bid")}</div>
          </div>
        </div>
      </section>
    `;
  }

  function chartCard() {
    return `
      <section class="card">
        <h2>Price Chart & Actions</h2>
        <div class="chart-wrap">
          <div id="candle-chart"></div>
        </div>
      </section>
    `;
  }

  function rightPanel(products, agents) {
    return `
      <section class="card right-panel">
        <div class="status-row">
          <span class="status-pill" id="market-pill">Market: idle</span>
          <span class="status-pill" id="agent-pill">Agent: idle</span>
        </div>

        <div class="control-group">
          <label for="product-select">Product</label>
          <select id="product-select">
            ${products.map((p) => `<option ${p === "SOLUSDT" ? "selected" : ""}>${p}</option>`).join("")}
          </select>
        </div>

        <div class="control-group">
          <label for="agent-select">Agent</label>
          <select id="agent-select">
            ${agents.map((a) => {
              const selected = a === "Mean Reversion" ? "selected" : "";
              if (a === "DQN") return `<option disabled title="Modèle non trouvé — placer models/dqn_agent.pth">DQN (modèle manquant)</option>`;
              if (a === "PPO") return `<option disabled title="TensorFlow non disponible sur Python 3.14">PPO (TF indisponible)</option>`;
              return `<option ${selected}>${a}</option>`;
            }).join("")}
          </select>
        </div>

        <button id="start-btn" class="start-btn">Start Simulation</button>
  <button id="stop-btn" class="stop-btn">Stop Simulation</button>
  <button id="clear-btn" class="clear-btn">🗑 Clear Session Data</button>

  <p class="helper-text" id="live-info">Ready.</p>
  <p class="helper-text error" id="live-error"></p>

        <div class="detail-panel">
          <h3 class="sub-title" id="detail-title">Inventory Details</h3>
          <div class="detail-grid" id="detail-grid"></div>
        </div>

        <h3 class="sub-title">Trade Log</h3>
        <div class="trade-log-wrap">
          <table class="log-table">
            <thead>
              <tr>
                <th>Time</th>
                <th>Action</th>
                <th>Qty</th>
                <th>Price</th>
                <th>Total</th>
              </tr>
            </thead>
            <tbody id="trade-log-body"></tbody>
          </table>
        </div>

      </section>
    `;
  }

  return {
    topbar,
    statCards,
    orderBookCard,
    chartCard,
    rightPanel,
  };
})();
