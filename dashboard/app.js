(function () {
  const C = window.MarketMindComponents;
  const state = JSON.parse(JSON.stringify(window.MarketMindData));
  let pollTimer = null;
  let activePanel = "inventory";
  let tvChart = null;
  let tvSeries = null;
  let isSwitchingSymbol = false;
  let lastMetrics = { cash: null, inventory: null, trade_count: 0, inventory_per_product: {} };
  let lastHistory = { trades: 0, buys: 0, sells: 0, net: 0, sell_ratio: 0 };

  function renderLayout() {
    const app = document.getElementById("app");
    app.innerHTML = `
      <div class="app-shell">
        ${C.topbar()}
        <main class="dashboard-grid">
          <section class="main-content">
            ${C.statCards()}
            <div class="top-cards">
              ${C.orderBookCard(state.orderBook)}
              ${C.chartCard()}
            </div>
            ${C.pnlCard()}
          </section>

          ${C.rightPanel(state.products, state.agents)}
        </main>
      </div>
    `;
  }

  function initTVChart() {
    const container = document.getElementById("candle-chart");
    if (!container || tvChart || !window.LightweightCharts) return;

    tvChart = LightweightCharts.createChart(container, {
      width: container.clientWidth,
      height: container.clientHeight,
      layout: {
        background: { type: "solid", color: "#0f1d32" },
        textColor: "#9fb3d1",
      },
      grid: {
        vertLines: { color: "rgba(159,179,209,0.08)" },
        horzLines: { color: "rgba(159,179,209,0.08)" },
      },
      rightPriceScale: { borderColor: "rgba(159,179,209,0.2)" },
      timeScale: {
        borderColor: "rgba(159,179,209,0.2)",
        timeVisible: true,
        secondsVisible: false,
      },
      crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    });

    tvSeries = tvChart.addCandlestickSeries({
      upColor: "#26a69a",
      downColor: "#ef5350",
      borderUpColor: "#26a69a",
      borderDownColor: "#ef5350",
      wickUpColor: "#26a69a",
      wickDownColor: "#ef5350",
    });

    new ResizeObserver(() => {
      if (tvChart && container.clientWidth > 0)
        tvChart.resize(container.clientWidth, container.clientHeight);
    }).observe(container);
  }

  function updateTVChart(candles) {
    if (!tvSeries) return;
    if (!Array.isArray(candles) || candles.length === 0) {
      tvSeries.setData([]);
      return;
    }

    const now = new Date();
    const seen = new Set();
    const data = [];
    for (const c of candles) {
      const [h, m, s] = c.t.split(":").map(Number);
      const d = new Date(now);
      d.setHours(h, m, s, 0);
      const ts = Math.floor(d.getTime() / 1000);
      if (seen.has(ts)) continue;
      seen.add(ts);
      data.push({ time: ts, open: c.o, high: c.h, low: c.l, close: c.c });
    }

    tvSeries.setData(data);
    tvChart.timeScale().fitContent();
  }

  function setInfo(message) {
    const el = document.getElementById("live-info");
    if (el) el.textContent = message;
  }

  function setError(message) {
    const el = document.getElementById("live-error");
    if (el) el.textContent = message || "";
  }

  function updateStatusPill(id, running, label) {
    const pill = document.getElementById(id);
    if (!pill) return;
    pill.textContent = `${label}: ${running ? "running" : "idle"}`;
    pill.classList.toggle("running", Boolean(running));
  }

  function setSwitchingState(enabled) {
    isSwitchingSymbol = enabled;

    const productSelect = document.getElementById("product-select");
    const agentSelect = document.getElementById("agent-select");
    const startBtn = document.getElementById("start-btn");
    const stopBtn = document.getElementById("stop-btn");
    const marketPill = document.getElementById("market-pill");

    if (productSelect) productSelect.disabled = enabled;
    if (agentSelect) agentSelect.disabled = enabled;
    if (startBtn) startBtn.disabled = enabled;
    if (stopBtn) stopBtn.disabled = enabled;

    if (marketPill) {
      marketPill.classList.toggle("switching", enabled);
      if (enabled) {
        marketPill.textContent = "Market: switching...";
      }
    }
  }

  function renderOrderBookLevels(payload) {
    const asksEl = document.getElementById("orderbook-asks");
    const bidsEl = document.getElementById("orderbook-bids");

    const asks = Array.isArray(payload?.asks)
      ? payload.asks.slice(0, 8).map((row) => ({
          price: Number(row.price) / 100,
          quantity: Number(row.quantity) / 100,
        }))
      : [];
    const bids = Array.isArray(payload?.bids)
      ? payload.bids.slice(0, 8).map((row) => ({
          price: Number(row.price) / 100,
          quantity: Number(row.quantity) / 100,
        }))
      : [];

    state.orderBook.asks = asks;
    state.orderBook.bids = bids;

    const renderRows = (rows, side) => {
      if (!rows.length) return '<div class="orderbook-empty">Waiting for live data...</div>';
      return rows
        .map(
          (row) => `
            <div class="orderbook-row ${side}">
              <span class="price">${row.price.toFixed(2)}</span>
              <span class="qty">${row.quantity.toFixed(4)}</span>
            </div>
          `
        )
        .join("");
    };

    if (asksEl) asksEl.innerHTML = renderRows(asks, "ask");
    if (bidsEl) bidsEl.innerHTML = renderRows(bids, "bid");
  }

  function renderTradeLog() {
    const tbody = document.getElementById("trade-log-body");
    if (!tbody) return;
    tbody.innerHTML = state.tradeLog
      .slice()
      .reverse()
      .map((row) => `
        <tr>
          <td>${row.time}</td>
          <td class="${row.action === "Buy" ? "log-buy" : "log-sell"}">${row.action}</td>
          <td>${row.qty != null ? row.qty.toFixed(4) : "--"}</td>
          <td>${row.price != null ? "$" + row.price.toFixed(2) : "--"}</td>
          <td class="${row.result >= 0 ? "pos" : "neg"}">${row.result >= 0 ? "+" : "-"}$${Math.abs(row.result).toFixed(2)}</td>
        </tr>
      `)
      .join("");
  }

  function renderSidebarMetrics(metrics) {
    const inventory = document.getElementById("side-inventory");
    const cash = document.getElementById("side-cash");
    const history = document.getElementById("side-history");

    // Affichage robuste : valeurs par défaut claires
    if (inventory) {
      if (metrics.inventory == null || isNaN(metrics.inventory)) {
        inventory.textContent = "--";
        inventory.classList.add("disabled-metric");
      } else {
        inventory.textContent = Number(metrics.inventory).toFixed(2);
        inventory.classList.remove("disabled-metric");
      }
    }
    if (cash) {
      if (metrics.cash == null || isNaN(metrics.cash)) {
        cash.textContent = "--";
        cash.classList.add("disabled-metric");
      } else {
        cash.textContent = `$${Number(metrics.cash).toFixed(2)}`;
        cash.classList.remove("disabled-metric");
      }
    }
    if (history) {
      if (metrics.trade_count == null || isNaN(metrics.trade_count) || metrics.trade_count === 0) {
        history.textContent = "--";
        history.classList.add("disabled-metric");
      } else {
        history.textContent = `${metrics.trade_count}`;
        history.classList.remove("disabled-metric");
      }
    }
  }

  function renderPnlChart(series) {
    const container = document.getElementById("pnl-chart");
    const valueEl = document.getElementById("pnl-value");
    if (!container) return;

    const last = series.length ? series[series.length - 1] : 0;
    if (valueEl) {
      valueEl.textContent = (last >= 0 ? "+" : "") + "$" + Math.abs(last).toFixed(2);
      valueEl.className = "pnl-value " + (last >= 0 ? "pos" : "neg");
    }

    if (!series.length) {
      container.innerHTML = '<div class="pnl-empty">En attente de données...</div>';
      return;
    }

    const W = 400;
    const H = 80;
    const px = 4;
    const min = Math.min(0, ...series);
    const max = Math.max(0, ...series);
    const range = max - min || 1;
    const xOf = (i) => px + (i / Math.max(1, series.length - 1)) * (W - 2 * px);
    const yOf = (v) => px + (H - 2 * px) - ((v - min) / range) * (H - 2 * px);
    const pts = series.map((v, i) => `${xOf(i).toFixed(1)},${yOf(v).toFixed(1)}`).join(" ");
    const zero = yOf(0).toFixed(1);
    const color = last >= 0 ? "#2ecc71" : "#e74c3c";
    const fillPts = `${xOf(0).toFixed(1)},${zero} ${pts} ${xOf(series.length - 1).toFixed(1)},${zero}`;

    container.innerHTML = `
      <svg width="100%" height="${H}" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
        <defs>
          <linearGradient id="pnl-grad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="${color}" stop-opacity="0.28"/>
            <stop offset="100%" stop-color="${color}" stop-opacity="0.02"/>
          </linearGradient>
        </defs>
        <polygon points="${fillPts}" fill="url(#pnl-grad)"/>
        <line x1="${px}" y1="${zero}" x2="${W - px}" y2="${zero}"
              stroke="rgba(159,179,209,0.25)" stroke-width="1" stroke-dasharray="4,3"/>
        <polyline points="${pts}" fill="none" stroke="${color}" stroke-width="2" stroke-linejoin="round"/>
        <circle cx="${xOf(series.length - 1).toFixed(1)}" cy="${yOf(last).toFixed(1)}" r="3.5" fill="${color}"/>
      </svg>`;
  }

  function renderPnlStats(history) {
    const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
    set("stat-winrate",  history.win_rate  != null ? `${history.win_rate.toFixed(1)}%` : "--");
    set("stat-avgprofit", history.avg_profit != null && history.avg_profit !== 0 ? `+$${history.avg_profit.toFixed(2)}` : "--");
    set("stat-avgloss",  history.avg_loss  != null && history.avg_loss  !== 0 ? `$${history.avg_loss.toFixed(2)}`  : "--");
    set("stat-trades",   history.trades    != null ? `${history.trades}` : "--");
  }

  function renderInventoryBar(inventory) {
    const bar = document.getElementById("inv-risk-bar");
    const pct = document.getElementById("inv-risk-pct");
    if (!bar || !pct) return;
    const abs = Math.abs(inventory || 0);
    const MAX_INV = 10;
    const ratio = Math.min(1, abs / MAX_INV);
    const pctVal = Math.round(ratio * 100);
    bar.style.width = `${pctVal}%`;
    bar.className = "inv-risk-bar " + (ratio > 0.66 ? "danger" : ratio > 0.33 ? "warn" : "safe");
    pct.textContent = `${pctVal}%`;
  }

  function renderLastDecision(decision) {
    const badge = document.getElementById("decision-badge");
    const timeEl = document.getElementById("decision-time");
    if (!badge) return;
    const action = decision?.action || "—";
    badge.textContent = action;
    badge.className = "decision-badge " + (
      action === "BUY"  ? "buy"  :
      action === "SELL" ? "sell" : "hold"
    );
    if (timeEl) timeEl.textContent = (decision?.time && decision.time !== "—") ? decision.time : "";
  }

  function renderPortfolio(midPrice) {
    const tbody = document.getElementById("portfolio-body");
    if (!tbody) return;
    const invMap = lastMetrics.inventory_per_product || {};
    const entries = Object.entries(invMap).filter(([, qty]) => qty != null && qty !== 0);
    if (entries.length === 0) {
      tbody.innerHTML = `<tr><td colspan="4" class="portfolio-empty">No open positions</td></tr>`;
      return;
    }
    const currentProduct = document.getElementById("product-select")?.value || "";
    tbody.innerHTML = entries.map(([product, qty]) => {
      const asset = product.replace(/USDT$|BUSD$|BTC$|ETH$|BNB$/, "") || product;
      const isCurrent = product === currentProduct;
      const price = isCurrent && midPrice ? midPrice : null;
      const value = price ? (qty * price).toFixed(2) : "--";
      const priceStr = price ? price.toFixed(2) : "--";
      return `<tr class="${isCurrent ? "portfolio-row-active" : ""}">
        <td>${asset}</td>
        <td>${Number(qty).toFixed(4)}</td>
        <td>${priceStr}</td>
        <td>${value !== "--" ? "$" + value : "--"}</td>
      </tr>`;
    }).join("");
  }

  function renderDetailPanel() {
    const title = document.getElementById("detail-title");
    const grid = document.getElementById("detail-grid");
    if (!title || !grid) return;

    const fmt = (v) => (v == null || isNaN(v)) ? "--" : Number(v).toFixed(2);

    // Derive base asset from selected product (SOLUSDT → SOL, ETHUSDT → ETH, etc.)
    const product = document.getElementById("product-select")?.value || "SOLUSDT";
    const baseAsset = product.replace(/USDT$|BUSD$|BTC$|ETH$|BNB$/, "") || product;

    // Build per-product inventory rows
    const invMap = lastMetrics.inventory_per_product || {};
    const invItems = Object.entries(invMap)
      .filter(([, qty]) => qty != null && qty !== 0)
      .map(([prod, qty]) => {
        const asset = prod.replace(/USDT$|BUSD$|BTC$|ETH$|BNB$/, "") || prod;
        const current = prod === product;
        return [`${asset}${current ? " ◀" : ""}`, Number(qty).toFixed(4)];
      });
    if (invItems.length === 0) invItems.push(["No positions", "—"]);

    const panels = {
      inventory: {
        title: "Positions by Product",
        items: invItems,
      },
      cash: {
        title: "Cash Details",
        items: [
          ["USDT Balance", (lastMetrics.cash != null && !isNaN(lastMetrics.cash)) ? `$${fmt(lastMetrics.cash)}` : "--"],
          ["Net PnL", (lastHistory.net != null && !isNaN(lastHistory.net)) ? `${lastHistory.net >= 0 ? "+" : "-"}$${Math.abs(lastHistory.net).toFixed(2)}` : "--"],
          ["Sell Ratio", (lastHistory.sell_ratio != null && !isNaN(lastHistory.sell_ratio)) ? `${lastHistory.sell_ratio.toFixed(1)}%` : "--"],
          ["Trades", (lastHistory.trades != null && !isNaN(lastHistory.trades)) ? `${lastHistory.trades}` : "--"],
        ],
      },
      history: {
        title: "History Details",
        items: [
          ["Total Trades", (lastHistory.trades != null && !isNaN(lastHistory.trades)) ? `${lastHistory.trades}` : "--"],
          ["Buys", (lastHistory.buys != null && !isNaN(lastHistory.buys)) ? `${lastHistory.buys}` : "--"],
          ["Sells", (lastHistory.sells != null && !isNaN(lastHistory.sells)) ? `${lastHistory.sells}` : "--"],
          ["Net Cash Flow", (lastHistory.net != null && !isNaN(lastHistory.net)) ? `${lastHistory.net >= 0 ? "+" : "-"}$${Math.abs(lastHistory.net).toFixed(2)}` : "--"],
        ],
      },
    };

    const panel = panels[activePanel] || panels.inventory;
    title.textContent = panel.title;
    // Masquer dynamiquement les lignes non pertinentes (valeur '--')
    grid.innerHTML = panel.items
      .filter(([_, v]) => v !== "--")
      .map(([k, v]) => `<div class="detail-item"><span class="k">${k}</span><span class="v">${v}</span></div>`)
      .join("");
  }

  function applyMarketState(payload) {
    if (!payload || !payload.ok) {
      state.orderBook.ask = null;
      state.orderBook.bid = null;
      renderOrderBookLevels({ asks: [], bids: [] });

      const askBadge = document.querySelector(".badge.ask");
      const bidBadge = document.querySelector(".badge.bid");
      const topbarPrice = document.getElementById("topbar-price");
      const spreadBadge = document.getElementById("spread-badge");
      if (askBadge) askBadge.textContent = "Ask --";
      if (bidBadge) bidBadge.textContent = "Bid --";
      if (topbarPrice) topbarPrice.textContent = "--";
      if (spreadBadge) spreadBadge.textContent = "Spread --";
      return;
    }
    const ask = payload.ask_price ? payload.ask_price / 100 : null;
    const bid = payload.bid_price ? payload.bid_price / 100 : null;

    if (ask && bid) {
      state.orderBook.ask = ask;
      state.orderBook.bid = bid;
      const askBadge = document.querySelector(".badge.ask");
      const bidBadge = document.querySelector(".badge.bid");
      if (askBadge) askBadge.textContent = `Ask ${ask.toFixed(2)}`;
      if (bidBadge) bidBadge.textContent = `Bid ${bid.toFixed(2)}`;

      const mid = (ask + bid) / 2;
      const spread = (ask - bid).toFixed(2);
      const topbarPrice = document.getElementById("topbar-price");
      if (topbarPrice) topbarPrice.textContent = `$${mid.toFixed(2)}`;
      const spreadBadge = document.getElementById("spread-badge");
      if (spreadBadge) spreadBadge.textContent = `Spread $${spread}`;
    }

    renderOrderBookLevels(payload);

    if (payload.symbol) {
      const subtitle = document.querySelector(".brand-text p");
      if (subtitle) subtitle.textContent = `Live Testnet Trading • ${payload.symbol}`;
    }
  }

  async function loadLiveData() {
    try {
      const res = await fetch("/api/snapshot");
      const snap = await res.json();

      const { market, trades, status, metrics, chart, history, performance, last_decision, username } = snap;

      if (username) {
        const el = document.getElementById("topbar-username");
        if (el) el.textContent = username;
      }

      applyMarketState(market);
      lastMetrics = { ...metrics, _mid: market?.mid_price ? market.mid_price / 100 : null };
      lastHistory = history;

      const product = document.getElementById("product-select")?.value || "SOLUSDT";
      const baseAsset = product.replace(/USDT$|BUSD$|BTC$|ETH$|BNB$/, "") || product;
      const invLabel = document.getElementById("inventory-label");
      if (invLabel) invLabel.textContent = `Inventory (${baseAsset})`;

      renderSidebarMetrics(metrics);
      renderDetailPanel();
      renderPortfolio(market?.mid_price ? market.mid_price / 100 : null);
      renderPnlChart(performance?.series || []);
      renderPnlStats(history);
      renderInventoryBar(metrics?.inventory);
      renderLastDecision(last_decision);

      state.tradeLog = Array.isArray(trades) ? trades : [];
      renderTradeLog();

      if (Array.isArray(chart.candles) && chart.candles.length) {
        state.candles = chart.candles.map((c) => ({ o: c.o, h: c.h, l: c.l, c: c.c, t: c.t }));
        updateTVChart(state.candles);
      } else {
        state.candles = [];
        updateTVChart([]);
      }

      const btn = document.getElementById("start-btn");
      const stopBtn = document.getElementById("stop-btn");
      const anyRunning = status.market_running || status.agent_running;

      if (btn) {
        btn.textContent = anyRunning ? "Running..." : "Start Simulation";
        if (!anyRunning) btn.disabled = false;
      }
      if (stopBtn) stopBtn.disabled = !anyRunning;

      if (!isSwitchingSymbol) updateStatusPill("market-pill", status.market_running, "Market");
      updateStatusPill("agent-pill", status.agent_running, "Agent");

      if (status.started_at) setInfo(`Last start: ${status.started_at.replace("T", " ")}`);
      setError(status.last_error || "");
    } catch (err) {
      setInfo("Backend API unavailable.");
      console.error("loadLiveData error:", err);
    }
  }

  function openPortfolioModal() {
    let modal = document.getElementById("portfolio-modal");
    if (!modal) {
      modal = document.createElement("div");
      modal.id = "portfolio-modal";
      modal.className = "portfolio-modal-overlay";
      modal.innerHTML = `
        <div class="portfolio-modal">
          <div class="portfolio-modal-header">
            <h2>Portfolio — All Positions</h2>
            <button class="portfolio-modal-close" id="portfolio-close">✕</button>
          </div>
          <table class="portfolio-table">
            <thead><tr><th>Asset</th><th>Quantity</th><th>Mid Price</th><th>Value (USDT)</th></tr></thead>
            <tbody id="portfolio-modal-body"></tbody>
          </table>
        </div>`;
      document.body.appendChild(modal);
      document.getElementById("portfolio-close").addEventListener("click", () => modal.remove());
      modal.addEventListener("click", (e) => { if (e.target === modal) modal.remove(); });
    }
    const invMap = lastMetrics.inventory_per_product || {};
    const currentProduct = document.getElementById("product-select")?.value || "";
    const entries = Object.entries(invMap).filter(([, qty]) => qty != null && qty !== 0);
    const tbody = document.getElementById("portfolio-modal-body");
    if (entries.length === 0) {
      tbody.innerHTML = `<tr><td colspan="4" class="portfolio-empty">No open positions</td></tr>`;
    } else {
      tbody.innerHTML = entries.map(([product, qty]) => {
        const asset = product.replace(/USDT$|BUSD$|BTC$|ETH$|BNB$/, "") || product;
        const isCurrent = product === currentProduct;
        const mid = isCurrent ? (lastMetrics._mid || null) : null;
        const value = mid ? "$" + (qty * mid).toFixed(2) : "--";
        const priceStr = mid ? mid.toFixed(2) : "--";
        return `<tr class="${isCurrent ? "portfolio-row-active" : ""}">
          <td>${asset}${isCurrent ? " <span class='portfolio-current-tag'>active</span>" : ""}</td>
          <td>${Number(qty).toFixed(4)}</td>
          <td>${priceStr}</td>
          <td>${value}</td>
        </tr>`;
      }).join("");
    }
    modal.style.display = "flex";
  }

  function bindSidebar() {
    const items = document.querySelectorAll(".stat-card");
    items.forEach((item) => {
      item.addEventListener("click", () => {
        items.forEach((it) => it.classList.remove("active"));
        item.classList.add("active");
        activePanel = item.getAttribute("data-panel") || "inventory";
        if (activePanel === "inventory") {
          openPortfolioModal();
        } else {
          renderDetailPanel();
        }
      });
    });
  }

  function bindControls() {
    const btn = document.getElementById("start-btn");
    const stopBtn = document.getElementById("stop-btn");
    const productSelect = document.getElementById("product-select");
    const agentSelect = document.getElementById("agent-select");

    async function startSimulation(product, agent) {
      const response = await fetch("/api/start_simulation", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ product, agent }),
      });

      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.message || "Failed to start simulation");
      }
    }

    if (productSelect) {
      productSelect.addEventListener("change", async () => {
        const subtitle = document.querySelector(".brand-text p");
        if (subtitle) subtitle.textContent = `Live Testnet Trading • ${productSelect.value}`;

        try {
          const statusRes = await fetch("/api/status");
          if (!statusRes.ok) return;
          const status = await statusRes.json();

          // If running, apply symbol change immediately by restarting with new product.
          if (status.market_running || status.agent_running) {
            const selectedProduct = productSelect.value || "SOLUSDT";
            const selectedAgent = agentSelect?.value || status.agent || "Mean Reversion";
            setInfo(`Switching symbol to ${selectedProduct}...`);
            setError("");
            setSwitchingState(true);
            await startSimulation(selectedProduct, selectedAgent);
            // Attendre que le backend ait bien changé de produit
            let tries = 0;
            while (tries < 10) {
              await new Promise((r) => setTimeout(r, 400));
              const marketRes = await fetch("/api/market_state");
              if (marketRes.ok) {
                const market = await marketRes.json();
                if (market.symbol === selectedProduct) break;
              }
              tries++;
            }
            // Ajout d'un délai pour laisser le backend écrire le nouvel état
            await new Promise((r) => setTimeout(r, 600));
            setInfo(`Symbol switched to ${selectedProduct}.`);
            await loadLiveData();
          } else {
            // Simulation pas lancée — démarrer quand même le marché pour avoir les prix live
            const selectedProduct = productSelect.value || "SOLUSDT";
            setInfo(`Loading market data for ${selectedProduct}...`);
            setSwitchingState(true);
            await fetch("/api/start_market", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ product: selectedProduct }),
            });
            await new Promise((r) => setTimeout(r, 1200));
            setInfo(`Showing live prices for ${selectedProduct}.`);
            await loadLiveData();
          }
        } catch (_) {
          setError("Failed to switch symbol while running.");
        } finally {
          setSwitchingState(false);
        }
      });
    }

    if (agentSelect) {
      agentSelect.addEventListener("change", async () => {
        try {
          const statusRes = await fetch("/api/status");
          if (!statusRes.ok) return;
          const status = await statusRes.json();
          if (status.market_running || status.agent_running) {
            const selectedProduct = productSelect?.value || status.product || "SOLUSDT";
            const selectedAgent = agentSelect.value;
            setInfo(`Switching agent to ${selectedAgent}...`);
            setError("");
            setSwitchingState(true);
            await startSimulation(selectedProduct, selectedAgent);
            await new Promise((r) => setTimeout(r, 800));
            setInfo(`Agent switched to ${selectedAgent}.`);
            await loadLiveData();
          }
        } catch (_) {
          setError("Failed to switch agent while running.");
        } finally {
          setSwitchingState(false);
        }
      });
    }

    if (btn) {
      btn.addEventListener("click", async () => {
        btn.disabled = true;
        btn.textContent = "Running...";
        setError("");
        setInfo("Starting simulation...");

        try {
          await startSimulation(productSelect?.value || "SOLUSDT", agentSelect?.value || "Mean Reversion");
          setInfo(`Simulation started: ${productSelect?.value || "SOLUSDT"} / ${agentSelect?.value || "Mean Reversion"}`);
          loadLiveData();
        } catch (err) {
          btn.disabled = false;
          btn.textContent = "Start Simulation";
          setError(err.message || "Could not start simulation from backend.");
        }
      });
    }

    if (stopBtn) {
      stopBtn.addEventListener("click", async () => {
        stopBtn.disabled = true;
        setInfo("Stopping simulation...");
        setError("");
        try {
          const response = await fetch("/api/stop_simulation", { method: "POST" });
          if (!response.ok) throw new Error("Failed to stop simulation");
          setInfo("Simulation stopped.");
        } catch (_) {
          setError("Failed to stop simulation.");
        } finally {
          loadLiveData();
        }
      });
    }

    const clearBtn = document.getElementById("clear-btn");
    if (clearBtn) {
      clearBtn.addEventListener("click", async () => {
        if (!confirm("Clear all session data? (logs, chart, performance)")) return;
        clearBtn.disabled = true;
        clearBtn.textContent = "Clearing...";
        try {
          const res = await fetch("/api/clear_logs", { method: "POST" });
          const data = await res.json();
          if (data.ok) {
            setInfo("Session data cleared. Waiting for backend reset...");
            state.candles = [];
            state.tradeLog = [];
            renderTradeLog();
            updateTVChart([]);
            // Poll /api/reset_status jusqu'à ce que reset soit terminé
            let tries = 0;
            while (tries < 50) { // max 5s
              const statusRes = await fetch("/api/reset_status");
              const status = await statusRes.json();
              if (!status.reset) break;
              await new Promise((r) => setTimeout(r, 100));
              tries++;
            }
            await loadLiveData();
          } else {
            setError("Failed to clear logs.");
          }
        } catch (_) {
          setError("Failed to clear logs.");
        } finally {
          clearBtn.disabled = false;
          clearBtn.textContent = "🗑 Clear Session Data";
        }
      });
    }
  }

  async function logout() {
    try {
      await fetch("/api/logout", { method: "POST" });
    } catch (_) {}
    window.location.replace("/login");
  }

  function init() {
    renderLayout();
    initTVChart();
    renderTradeLog();
    bindControls();
    bindSidebar();
    renderDetailPanel();

    const logoutBtn = document.getElementById("logout-btn");
    if (logoutBtn) logoutBtn.addEventListener("click", logout);

    // Démarrer le marché par défaut au chargement pour avoir les prix live immédiatement
    const defaultProduct = document.getElementById("product-select")?.value || "SOLUSDT";
    fetch("/api/start_market", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ product: defaultProduct }),
    }).catch(() => {});

    loadLiveData();
    pollTimer = setInterval(loadLiveData, 2000);

    window.addEventListener("beforeunload", () => {
      if (pollTimer) clearInterval(pollTimer);
    });
  }

  init();
})();
