(function () {
  const C = window.MarketMindComponents;
  const state = JSON.parse(JSON.stringify(window.MarketMindData));
  let pollTimer = null;
  let recentLogs = { agent: [], market: [] };
  let activeLogTab = "agent";
  let activePanel = "inventory";
  let isSwitchingSymbol = false;
  let lastMetrics = { cash: null, inventory: null, trade_count: 0 };
  let lastHistory = { trades: 0, wins: 0, losses: 0, net: 0, win_rate: 0 };

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
            ${C.performanceCard()}
          </section>

          ${C.rightPanel(state.products, state.agents)}
        </main>
        <div class="logs-wrapper">
          ${C.logsSection()}
        </div>
      </div>
    `;
  }

  function drawCandleChart(svg, candles, axisValues) {
    if (!svg) return;
    if (!Array.isArray(candles) || candles.length === 0) {
      svg.innerHTML = "";
      return;
    }

    const W = 780;
    const H = 290;
    const left = 14;
    const right = 58;
    const top = 10;
    const innerW = W - left - right;
    const innerH = H - top - 20;

    const highs = candles.map((c) => c.h);
    const lows = candles.map((c) => c.l);
    const min = Math.min(...lows) - 0.1;
    const max = Math.max(...highs) + 0.1;
    const range = (max - min) || 1;
    const y = (price) => top + ((max - price) / range) * innerH;
    const xStep = innerW / candles.length;

    let html = "";
    for (let i = 0; i <= 6; i++) {
      const gy = top + (innerH / 6) * i;
      html += `<line x1="${left}" y1="${gy}" x2="${W - right}" y2="${gy}" stroke="rgba(159,179,209,0.16)" stroke-width="1"/>`;
    }

    axisValues.forEach((value, idx) => {
      const yy = top + (innerH / Math.max(1, axisValues.length - 1)) * idx;
      html += `<text x="${W - right + 8}" y="${yy + 4}" fill="#9fb3d1" font-size="10">${value.toFixed(1)}</text>`;
    });

    candles.forEach((c, i) => {
      const cx = left + xStep * i + xStep * 0.5;
      const isUp = c.c >= c.o;
      const color = isUp ? "#2ecc71" : "#ff5a5f";
      const openY = y(c.o);
      const closeY = y(c.c);
      const bodyY = Math.min(openY, closeY);
      const bodyH = Math.max(3, Math.abs(closeY - openY));

      html += `<line x1="${cx}" y1="${y(c.h)}" x2="${cx}" y2="${y(c.l)}" stroke="${color}" stroke-width="1.6"/>`;
      html += `<rect x="${cx - 7}" y="${bodyY}" width="14" height="${bodyH}" fill="${color}" opacity="0.92" rx="2"/>`;
    });

    svg.innerHTML = html;
  }

  function drawPerformanceChart(svg, series, ticks) {
    if (!svg) return;
    if (!Array.isArray(series) || series.length === 0) {
      svg.innerHTML = "";
      return;
    }

    const W = 1180;
    const H = 220;
    const left = 12;
    const right = 12;
    const top = 10;
    const innerW = W - left - right;
    const innerH = H - top - 30;

    const min = Math.min(...series) - 10;
    const max = Math.max(...series) + 10;
    const range = (max - min) || 1;
    const xStep = series.length > 1 ? innerW / (series.length - 1) : innerW;

    const xy = series.map((v, i) => {
      const x = left + i * xStep;
      const y = top + ((max - v) / range) * innerH;
      return [x, y];
    });

    let html = "";
    for (let i = 0; i <= 5; i++) {
      const gy = top + (innerH / 5) * i;
      html += `<line x1="${left}" y1="${gy}" x2="${W - right}" y2="${gy}" stroke="rgba(159,179,209,0.14)" stroke-width="1"/>`;
    }

    ticks.forEach((t, i) => {
      const x = left + (innerW / Math.max(1, ticks.length - 1)) * i;
      html += `<text x="${x - 14}" y="${H - 10}" fill="#9fb3d1" font-size="11">${t}</text>`;
    });

    const linePoints = xy.map((p) => `${p[0]},${p[1]}`).join(" ");
    const areaPoints = `${left},${top + innerH} ${linePoints} ${W - right},${top + innerH}`;

    html += `<polygon points="${areaPoints}" fill="rgba(74,163,255,0.12)"/>`;
    html += `<polyline points="${linePoints}" fill="none" stroke="#4aa3ff" stroke-width="2.4"/>`;
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
          ["Win Rate", (lastHistory.win_rate != null && !isNaN(lastHistory.win_rate)) ? `${lastHistory.win_rate.toFixed(2)}%` : "--"],
          ["Trades", (lastHistory.trades != null && !isNaN(lastHistory.trades)) ? `${lastHistory.trades}` : "--"],
        ],
      },
      history: {
        title: "History Details",
        items: [
          ["Total Trades", (lastHistory.trades != null && !isNaN(lastHistory.trades)) ? `${lastHistory.trades}` : "--"],
          ["Wins", (lastHistory.wins != null && !isNaN(lastHistory.wins)) ? `${lastHistory.wins}` : "--"],
          ["Losses", (lastHistory.losses != null && !isNaN(lastHistory.losses)) ? `${lastHistory.losses}` : "--"],
          ["Net Result", (lastHistory.net != null && !isNaN(lastHistory.net)) ? `${lastHistory.net >= 0 ? "+" : "-"}$${Math.abs(lastHistory.net).toFixed(2)}` : "--"],
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

  function renderRecentLogs() {
    const logBox = document.getElementById("recent-logs");
    if (!logBox) return;
    const lines = recentLogs[activeLogTab] || [];
    logBox.textContent = lines.length ? lines.join("\n") : "No logs yet.";
    // Auto-scroll to bottom to show latest lines
    logBox.scrollTop = logBox.scrollHeight;

    const agentTab = document.getElementById("tab-agent");
    const marketTab = document.getElementById("tab-market");
    if (agentTab) agentTab.classList.toggle("active", activeLogTab === "agent");
    if (marketTab) marketTab.classList.toggle("active", activeLogTab === "market");
  }

  function renderChartAnnotations(markers, candles = state.candles) {
    const box = document.getElementById("chart-annotations");
    if (!box) return;
    box.innerHTML = "";

    const subset = (Array.isArray(markers) ? markers : []).slice(-4);
    const safeCandles = Array.isArray(candles) ? candles : [];
    if (!subset.length || !safeCandles.length) return;

    const highs = safeCandles.map((c) => c.h);
    const lows = safeCandles.map((c) => c.l);
    const min = Math.min(...lows);
    const max = Math.max(...highs);
    const range = (max - min) || 1;

    subset.forEach((marker, idx) => {
      const cls = marker.side === "sell" ? "sell" : "buy";
      const foundIndex = safeCandles.findIndex((c) => c.t === marker.time);
      const candleIndex = foundIndex >= 0 ? foundIndex : Math.max(0, safeCandles.length - 1 - idx);
      const candle = safeCandles[candleIndex] || safeCandles[safeCandles.length - 1];
      const x = safeCandles.length > 1 ? 6 + (candleIndex / (safeCandles.length - 1)) * 88 : 50;
      const price = cls === "sell" ? candle.h : candle.l;
      const y = 10 + ((max - price) / range) * 70;
      const el = document.createElement("div");
      el.className = `chart-annotation ${cls}`;
      el.style.left = `${x}%`;
      el.style.top = `${y}%`;
      el.textContent = marker.label || (cls === "sell" ? "Sell Order" : "Buy Order");
      box.appendChild(el);
    });
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
      if (subtitle) subtitle.textContent = `AI Trading Simulator • ${payload.symbol}`;
    }
  }

  async function loadLiveData() {
    try {
      const [marketRes, tradesRes, perfRes, statusRes, metricsRes, logsRes, chartRes, historyRes] = await Promise.all([
        fetch("/api/market_state"),
        fetch("/api/trade_log"),
        fetch("/api/performance"),
        fetch("/api/status"),
        fetch("/api/metrics"),
        fetch("/api/recent_logs"),
        fetch("/api/chart"),
        fetch("/api/history"),
      ]);

      const market = await marketRes.json();
      const trades = await tradesRes.json();
      const perf = await perfRes.json();
      const status = await statusRes.json();
      const metrics = await metricsRes.json();
      const logs = await logsRes.json();
      const chart = await chartRes.json();
      const history = await historyRes.json();

      applyMarketState(market);
      lastMetrics = metrics;
      lastHistory = history;

      renderSidebarMetrics(metrics);
      renderDetailPanel();

      state.tradeLog = Array.isArray(trades.rows) ? trades.rows : [];
      renderTradeLog();

      if (Array.isArray(perf.series) && perf.series.length) {
        state.performance.series = perf.series;
        if (Array.isArray(perf.ticks) && perf.ticks.length) {
          state.performance.ticks = perf.ticks;
        }
        drawPerformanceChart(document.getElementById("perf-chart"), state.performance.series, state.performance.ticks);
      } else {
        state.performance.series = [];
        state.performance.ticks = [];
        drawPerformanceChart(document.getElementById("perf-chart"), [], []);
      }

      if (Array.isArray(chart.candles) && chart.candles.length) {
        state.candles = chart.candles.map((c) => ({ o: c.o, h: c.h, l: c.l, c: c.c, t: c.t }));
        state.priceAxis = Array.isArray(chart.axis) && chart.axis.length ? chart.axis : state.priceAxis;
        drawCandleChart(document.getElementById("candle-chart"), state.candles, state.priceAxis);
        renderChartAnnotations(chart.markers, state.candles);

        const tag = document.getElementById("chart-tag");
        if (tag) {
          if (!state.candles.length) {
            tag.textContent = "Waiting for market data";
          } else if (chart.tag) {
            tag.textContent = chart.tag;
          }
        }
      } else {
        state.candles = [];
        drawCandleChart(document.getElementById("candle-chart"), [], []);
        renderChartAnnotations([], []);
        const tag = document.getElementById("chart-tag");
        if (tag) tag.textContent = "Waiting for market data";
      }

      recentLogs = {
        agent: Array.isArray(logs.agent) ? logs.agent.slice(-40) : [],
        market: Array.isArray(logs.market) ? logs.market.slice(-40) : [],
      };
      renderRecentLogs();

      const btn = document.getElementById("start-btn");
      const stopBtn = document.getElementById("stop-btn");
      const anyRunning = status.market_running || status.agent_running;

      if (btn) {
        if (anyRunning) {
          btn.textContent = "Running...";
        } else {
          btn.textContent = "Start Simulation";
          btn.disabled = false;
        }
      }
      if (stopBtn) stopBtn.disabled = !anyRunning;

      if (!isSwitchingSymbol) {
        updateStatusPill("market-pill", status.market_running, "Market");
      }
      updateStatusPill("agent-pill", status.agent_running, "Agent");

      if (status.started_at) setInfo(`Last start: ${status.started_at.replace("T", " ")}`);
      setError(status.last_error || "");
    } catch (_) {
      setInfo("Backend API unavailable – fallback mock mode.");
    }
  }

  function bindLogTabs() {
    const agentTab = document.getElementById("tab-agent");
    const marketTab = document.getElementById("tab-market");
    if (agentTab) {
      agentTab.addEventListener("click", () => {
        activeLogTab = "agent";
        renderRecentLogs();
      });
    }
    if (marketTab) {
      marketTab.addEventListener("click", () => {
        activeLogTab = "market";
        renderRecentLogs();
      });
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
        if (subtitle) subtitle.textContent = `AI Trading Simulator • ${productSelect.value}`;

        try {
          const statusRes = await fetch("/api/status");
          if (!statusRes.ok) return;
          const status = await statusRes.json();

          // If running, apply symbol change immediately by restarting with new product.
          if (status.market_running || status.agent_running) {
            const selectedProduct = productSelect.value || "SOLUSDT";
            const selectedAgent = agentSelect?.value || status.agent || "RL Agent";
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
          await startSimulation(productSelect?.value || "SOLUSDT", agentSelect?.value || "RL Agent");
          setInfo(`Simulation started: ${productSelect?.value || "SOLUSDT"} / ${agentSelect?.value || "RL Agent"}`);
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
            state.performance.series = [];
            state.candles = [];
            state.tradeLog = [];
            renderTradeLog();
            drawCandleChart(document.getElementById("candle-chart"), [], []);
            drawPerformanceChart(document.getElementById("perf-chart"), [], []);
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
    drawCandleChart(document.getElementById("candle-chart"), state.candles, state.priceAxis);
    drawPerformanceChart(document.getElementById("perf-chart"), state.performance.series, state.performance.ticks);
    renderTradeLog();
    bindControls();
    bindLogTabs();
    bindSidebar();
    renderRecentLogs();
    renderDetailPanel();

    loadLiveData();
    pollTimer = setInterval(loadLiveData, 2000);

    window.addEventListener("beforeunload", () => {
      if (pollTimer) clearInterval(pollTimer);
    });
  }

  init();
})();
