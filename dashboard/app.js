(function () {
  const C = window.MarketMindComponents;
  const state = JSON.parse(JSON.stringify(window.MarketMindData));
  let pollTimer = null;
  let activePanel = "inventory";
  let isSwitchingSymbol = false;
  let lastMetrics = { cash: null, inventory: null, trade_count: 0 };
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
          </section>

          ${C.rightPanel(state.products, state.agents)}
        </main>
      </div>
    `;
  }

  function drawCandleChart(svg, candles) {
    if (!svg) return;

    if (!Array.isArray(candles) || candles.length === 0) {
      svg.innerHTML = '<text x="390" y="148" fill="#9fb3d1" font-size="13" text-anchor="middle">Waiting for market data...</text>';
      return;
    }

    const W = 780, H = 290;
    const m = { top: 14, right: 68, bottom: 26, left: 10 };
    const cW = W - m.left - m.right;
    const cH = H - m.top - m.bottom;

    const dataMax = Math.max(...candles.map((c) => c.h));
    const dataMin = Math.min(...candles.map((c) => c.l));
    const priceSpan = dataMax - dataMin || 1;
    const vPad = priceSpan * 0.1;
    const yMax = dataMax + vPad;
    const yMin = dataMin - vPad;
    const ySpan = yMax - yMin;

    const toY = (p) => m.top + (1 - (p - yMin) / ySpan) * cH;
    const slot = cW / candles.length;
    const toX = (i) => m.left + slot * i + slot / 2;
    const barW = Math.max(2, Math.floor(slot * 0.55));

    let html = "";

    // Y grid + labels (6 lines)
    for (let i = 0; i <= 5; i++) {
      const price = yMax - (i / 5) * ySpan;
      const gy = toY(price);
      html += `<line x1="${m.left}" y1="${gy}" x2="${W - m.right}" y2="${gy}" stroke="rgba(159,179,209,0.13)" stroke-width="1"/>`;
      html += `<text x="${W - m.right + 6}" y="${gy}" fill="#9fb3d1" font-size="10" dominant-baseline="middle">${price.toFixed(2)}</text>`;
    }

    // X time labels — show at most 6 evenly distributed
    const labelStep = Math.max(1, Math.floor(candles.length / 6));
    candles.forEach((c, i) => {
      if (i % labelStep === 0 || i === candles.length - 1) {
        html += `<text x="${toX(i)}" y="${H - 4}" fill="#6d849c" font-size="9" text-anchor="middle">${c.t}</text>`;
      }
    });

    // Candles
    candles.forEach((c, i) => {
      const x = toX(i);
      const isUp = c.c >= c.o;
      const fill = isUp ? "#26a69a" : "#ef5350";
      const wickColor = isUp ? "#1a7a71" : "#b33c3c";
      const bodyTop = Math.min(toY(c.o), toY(c.c));
      const bodyH = Math.max(1.5, Math.abs(toY(c.c) - toY(c.o)));

      html += `<line x1="${x}" y1="${toY(c.h)}" x2="${x}" y2="${toY(c.l)}" stroke="${wickColor}" stroke-width="1.2"/>`;
      html += `<rect x="${x - barW / 2}" y="${bodyTop}" width="${barW}" height="${bodyH}" fill="${fill}" rx="1"/>`;
    });

    svg.innerHTML = html;
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

  function renderDetailPanel() {
    const title = document.getElementById("detail-title");
    const grid = document.getElementById("detail-grid");
    if (!title || !grid) return;

    const fmt = (v) => (v == null || isNaN(v)) ? "--" : Number(v).toFixed(2);

    // Derive base asset from selected product (SOLUSDT → SOL, ETHUSDT → ETH, etc.)
    const product = document.getElementById("product-select")?.value || "SOLUSDT";
    const baseAsset = product.replace(/USDT$|BUSD$|BTC$|ETH$|BNB$/, "") || product;

    const panels = {
      inventory: {
        title: "Inventory Details",
        items: [
          ["Asset", baseAsset],
          ["Quantity", fmt(lastMetrics.inventory)],
          ["Bid Value", (lastMetrics.inventory != null && state.orderBook.bid != null && !isNaN(lastMetrics.inventory) && !isNaN(state.orderBook.bid)) ? `$${fmt(lastMetrics.inventory * state.orderBook.bid)}` : "--"],
          ["Updated", new Date().toLocaleTimeString()],
        ],
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

      const { market, trades, status, metrics, chart, history } = snap;

      applyMarketState(market);
      lastMetrics = metrics;
      lastHistory = history;

      const product = document.getElementById("product-select")?.value || "SOLUSDT";
      const baseAsset = product.replace(/USDT$|BUSD$|BTC$|ETH$|BNB$/, "") || product;
      const invLabel = document.getElementById("inventory-label");
      if (invLabel) invLabel.textContent = `Inventory (${baseAsset})`;

      renderSidebarMetrics(metrics);
      renderDetailPanel();

      state.tradeLog = Array.isArray(trades) ? trades : [];
      renderTradeLog();

      if (Array.isArray(chart.candles) && chart.candles.length) {
        state.candles = chart.candles.map((c) => ({ o: c.o, h: c.h, l: c.l, c: c.c, t: c.t }));
        drawCandleChart(document.getElementById("candle-chart"), state.candles);
      } else {
        state.candles = [];
        drawCandleChart(document.getElementById("candle-chart"), []);
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
    } catch (_) {
      setInfo("Backend API unavailable – fallback mock mode.");
    }
  }

  function bindSidebar() {
    const items = document.querySelectorAll(".stat-card");
    items.forEach((item) => {
      item.addEventListener("click", () => {
        items.forEach((it) => it.classList.remove("active"));
        item.classList.add("active");
        activePanel = item.getAttribute("data-panel") || "inventory";
        renderDetailPanel();
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

      if (!response.ok) throw new Error("Failed to start simulation");
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
            // Ajout d'un délai même si pas de simulation
            await new Promise((r) => setTimeout(r, 600));
            await loadLiveData();
          }
        } catch (_) {
          setError("Failed to switch symbol while running.");
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
        } catch (_) {
          btn.disabled = false;
          btn.textContent = "Start Simulation";
          setError("Could not start simulation from backend.");
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
            drawCandleChart(document.getElementById("candle-chart"), []);
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

  function init() {
    renderLayout();
    drawCandleChart(document.getElementById("candle-chart"), state.candles);
    renderTradeLog();
    bindControls();
    bindSidebar();
    renderDetailPanel();

    loadLiveData();
    pollTimer = setInterval(loadLiveData, 2000);

    window.addEventListener("beforeunload", () => {
      if (pollTimer) clearInterval(pollTimer);
    });
  }

  init();
})();
