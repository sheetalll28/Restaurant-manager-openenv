const state = {
  taskId: null,
  latestObservation: null,
  tasks: [],
};

async function getJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const payload = await response.text();
    throw new Error(payload);
  }
  return response.json();
}

function formatJson(value) {
  return JSON.stringify(value, null, 2);
}

function byId(id) {
  return document.getElementById(id);
}

function renderTaskOptions(tasks) {
  const select = byId("taskSelect");
  select.innerHTML = "";
  tasks.forEach((task) => {
    const option = document.createElement("option");
    option.value = task.id;
    option.textContent = `${task.name} (${task.difficulty})`;
    select.appendChild(option);
  });
  if (tasks.length) {
    select.value = tasks[0].id;
  }
}

function renderTaskCards(tasks) {
  const host = byId("taskCards");
  host.innerHTML = "";
  tasks.forEach((task) => {
    const card = document.createElement("article");
    card.className = "task-card";
    if (task.id === state.taskId) {
      card.classList.add("active");
    }
    card.innerHTML = `
      <span>${task.id}</span>
      <strong>${task.name}</strong>
      <p>${task.description}</p>
      <p>Targets: profit ${task.targets.profit}, rating ${task.targets.rating}, service ${task.targets.service_rate}</p>
    `;
    host.appendChild(card);
  });
}

function renderSummary(observation) {
  const cards = [
    ["Step", `${observation.step}/${observation.total_steps}`],
    ["Rating", observation.customer_rating],
    ["Revenue", observation.revenue],
    ["Costs", observation.costs],
  ];
  const host = byId("summaryCards");
  host.innerHTML = "";
  cards.forEach(([label, value]) => {
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = `<span>${label}</span><strong>${value}</strong>`;
    host.appendChild(card);
  });
}

function renderInventory(observation) {
  const host = byId("inventoryView");
  host.innerHTML = "";
  observation.inventory.forEach((item) => {
    const row = document.createElement("div");
    row.className = "table-row";
    row.innerHTML = `<span>${item.name}</span><strong>${item.quantity} ${item.unit}</strong>`;
    host.appendChild(row);
  });
}

function buildCheckboxList(hostId, items, checkedField = "is_active") {
  const host = byId(hostId);
  host.innerHTML = "";
  items.forEach((item) => {
    const row = document.createElement("label");
    row.className = "checkbox-row";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.dataset.name = item.name;
    input.checked = Boolean(item[checkedField]);
    row.append(document.createTextNode(item.name), input);
    host.appendChild(row);
  });
}

function buildNumericList(hostId, items, valueField, step = "1") {
  const host = byId(hostId);
  host.innerHTML = "";
  items.forEach((item) => {
    const row = document.createElement("label");
    row.className = "control-row";
    const span = document.createElement("span");
    span.textContent = item.name;
    const input = document.createElement("input");
    input.type = "number";
    input.step = step;
    input.placeholder = `${item[valueField]}`;
    input.dataset.name = item.name;
    row.append(span, input);
    host.appendChild(row);
  });
}

function renderControls(observation) {
  buildCheckboxList("staffControls", observation.staff);
  buildCheckboxList("menuControls", observation.menu, "available");
  buildNumericList("priceControls", observation.menu, "price", "1");
  buildNumericList("reorderControls", observation.inventory, "quantity", "0.1");
}

function renderObservation(observation) {
  state.latestObservation = observation;
  renderSummary(observation);
  renderInventory(observation);
  renderControls(observation);
}

function collectCheckboxMap(hostId, originalItems, checkedField = "is_active") {
  const original = Object.fromEntries(originalItems.map((item) => [item.name, Boolean(item[checkedField])]));
  const result = {};
  byId(hostId).querySelectorAll("input[type='checkbox']").forEach((input) => {
    if (original[input.dataset.name] !== input.checked) {
      result[input.dataset.name] = input.checked;
    }
  });
  return result;
}

function collectNumberMap(hostId) {
  const result = {};
  byId(hostId).querySelectorAll("input[type='number']").forEach((input) => {
    if (input.value !== "") {
      result[input.dataset.name] = Number(input.value);
    }
  });
  return result;
}

function log(message, payload) {
  const view = byId("logView");
  const lines = [message];
  if (payload) {
    lines.push(formatJson(payload));
  }
  view.textContent = `${lines.join("\n")}\n\n${view.textContent}`.trim();
}

async function loadTasks() {
  const payload = await getJson("/tasks");
  state.tasks = payload.tasks;
  byId("scenarioCount").textContent = `${state.tasks.length} scenarios`;
  byId("taskMeta").textContent = `${state.tasks.length} scenarios loaded. Select one from the dropdown and reset to switch the active environment.`;
  renderTaskOptions(state.tasks);
  renderTaskCards(state.tasks);
}

async function resetTask() {
  const taskId = byId("taskSelect").value;
  const payload = await getJson("/reset", {
    method: "POST",
    body: JSON.stringify({ task_id: taskId }),
  });
  state.taskId = payload.task_id;
  const activeTask = state.tasks.find((task) => task.id === state.taskId);
  if (activeTask) {
    byId("taskMeta").textContent = `Active scenario: ${activeTask.name} (${activeTask.id})`;
  }
  renderObservation(payload.observation);
  renderTaskCards(state.tasks);
  byId("resultView").textContent = "Run the shift to see the final result.";
  log(`Reset ${taskId}`, payload.observation);
}

async function refreshState() {
  const payload = await getJson("/state");
  renderObservation(payload.observation);
  log(`Refreshed state for ${payload.task_id}`, payload.observation);
}

async function stepTask() {
  if (!state.latestObservation) {
    await resetTask();
  }
  const action = {
    staff_changes: collectCheckboxMap("staffControls", state.latestObservation.staff),
    menu_changes: collectCheckboxMap("menuControls", state.latestObservation.menu, "available"),
    price_adjustments: collectNumberMap("priceControls"),
    reorder_inventory: collectNumberMap("reorderControls"),
    promotion_active: byId("promotionToggle").checked,
  };
  const payload = await getJson("/step", {
    method: "POST",
    body: JSON.stringify(action),
  });
  renderObservation(payload.observation);
  log(`Step reward ${payload.reward}`, payload.info);
  if (payload.done) {
    byId("resultView").textContent = formatJson({
      final_score: payload.info.final_score,
      pillar_scores: payload.info.pillar_scores,
      shift_result: payload.info.shift_result,
    });
  }
}

async function fetchResult() {
  const payload = await getJson("/result");
  byId("resultView").textContent = formatJson(payload);
  log(`Fetched result for ${payload.task_id}`, payload);
}

async function bootstrap() {
  await loadTasks();
  await resetTask();

  byId("resetBtn").addEventListener("click", resetTask);
  byId("stepBtn").addEventListener("click", stepTask);
  byId("refreshBtn").addEventListener("click", refreshState);
  byId("resultBtn").addEventListener("click", fetchResult);
}

bootstrap().catch((error) => {
  byId("logView").textContent = error.message;
});
