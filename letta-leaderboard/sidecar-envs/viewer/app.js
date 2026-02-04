const state = {
  index: null,
  category: null,
  data: null,
  models: [],
  selectedModel: null,
  search: "",
  selectedId: null,
  loading: false,
  fetchGeneration: 0,
};

const el = (id) => document.getElementById(id);

const categoryCountEl = el("categoryCount");
const scenarioCountEl = el("scenarioCount");
const categoryTabsEl = el("categoryTabs");
const modelSelectEl = el("modelSelect");
const searchInputEl = el("searchInput");
const listMetaEl = el("listMeta");
const scenarioListEl = el("scenarioList");
const contextMetaEl = el("contextMeta");
const rubricMetaEl = el("rubricMeta");
const responseMetaEl = el("responseMeta");
const contextBodyEl = el("contextBody");
const rubricBodyEl = el("rubricBody");
const responseBodyEl = el("responseBody");
const judgeMetaEl = el("judgeMeta");
const judgeBodyEl = el("judgeBody");

function toTitleCase(text) {
  return text
    .split(/\s+/)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

function truncate(text, max) {
  if (!text) return "";
  const compact = text.replace(/\s+/g, " ").trim();
  if (compact.length <= max) return compact;
  return `${compact.slice(0, max)}…`;
}

function debounce(fn, ms) {
  let timeout;
  return (...args) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => fn(...args), ms);
  };
}

function setLoading(loading) {
  state.loading = loading;
  document.body.classList.toggle("is-loading", loading);
}

function showError(message) {
  contextBodyEl.innerHTML = "";
  rubricBodyEl.innerHTML = "";
  responseBodyEl.innerHTML = "";
  judgeBodyEl.innerHTML = "";

  const errorEl = document.createElement("div");
  errorEl.className = "error-message";
  errorEl.textContent = message;

  contextBodyEl.appendChild(errorEl.cloneNode(true));
  rubricBodyEl.appendChild(errorEl.cloneNode(true));
  judgeBodyEl.appendChild(errorEl.cloneNode(true));
  responseBodyEl.appendChild(errorEl);
}

function updateStats() {
  if (!state.index) return;
  categoryCountEl.textContent = state.index.categories.length;
  const total = state.index.categories.reduce(
    (sum, cat) => sum + cat.scenario_count,
    0
  );
  scenarioCountEl.textContent = total;
}

async function loadIndex() {
  setLoading(true);
  try {
    const response = await fetch("data/index.json");
    if (!response.ok) {
      throw new Error(`Failed to load index: ${response.status}`);
    }
    state.index = await response.json();
    updateStats();
    renderCategoryTabs();
    const first = state.index.categories[0];
    if (first) {
      await selectCategory(first.name);
    }
  } catch (err) {
    showError(`Failed to load data: ${err.message}`);
  } finally {
    setLoading(false);
  }
}

function renderCategoryTabs() {
  categoryTabsEl.innerHTML = "";
  state.index.categories.forEach((cat) => {
    const button = document.createElement("button");
    button.className = "tab";
    if (state.category && state.category.name === cat.name) {
      button.classList.add("active");
    }
    button.textContent = cat.display_name || toTitleCase(cat.name);
    button.addEventListener("click", () => selectCategory(cat.name));
    categoryTabsEl.appendChild(button);
  });
}

function deriveModels(data) {
  if (data.models && data.models.length) {
    return data.models.map((model) => ({
      model_name: model.model_name,
      avg_score: model.avg_score,
    }));
  }

  const seen = new Map();
  data.scenarios.forEach((scenario) => {
    (scenario.results || []).forEach((result) => {
      if (!seen.has(result.model_name)) {
        seen.set(result.model_name, {
          model_name: result.model_name,
          avg_score: null,
        });
      }
    });
  });

  return Array.from(seen.values()).sort((a, b) =>
    a.model_name.localeCompare(b.model_name)
  );
}

function renderModelSelect() {
  modelSelectEl.innerHTML = "";
  if (!state.models.length) {
    const option = document.createElement("option");
    option.textContent = "No model results";
    option.value = "";
    modelSelectEl.appendChild(option);
    modelSelectEl.disabled = true;
    return;
  }

  modelSelectEl.disabled = false;
  state.models.forEach((model) => {
    const option = document.createElement("option");
    option.value = model.model_name;
    const scoreText =
      typeof model.avg_score === "number"
        ? ` (${(model.avg_score * 100).toFixed(1)}%)`
        : "";
    option.textContent = `${model.model_name}${scoreText}`;
    if (model.model_name === state.selectedModel) {
      option.selected = true;
    }
    modelSelectEl.appendChild(option);
  });
}

async function selectCategory(name) {
  const cat = state.index.categories.find((item) => item.name === name);
  if (!cat) return;

  const generation = ++state.fetchGeneration;
  setLoading(true);
  try {
    const response = await fetch(cat.data_file);
    if (!response.ok) {
      throw new Error(`Failed to load category: ${response.status}`);
    }
    const data = await response.json();

    if (generation !== state.fetchGeneration) return;

    state.category = cat;
    state.data = data;
    state.models = deriveModels(data);
    state.selectedModel = state.models[0]?.model_name || null;
    state.search = "";
    searchInputEl.value = "";

    const sorted = getFilteredScenarios();
    state.selectedId = sorted[0]?.id || null;

    renderCategoryTabs();
    renderModelSelect();
    renderScenarioList();
    renderPanels();
  } catch (err) {
    showError(`Failed to load category "${name}": ${err.message}`);
  } finally {
    setLoading(false);
  }
}

function getFilteredScenarios() {
  if (!state.data) return [];
  const query = state.search.trim().toLowerCase();
  return state.data.scenarios
    .filter((scenario) => {
      if (!query) return true;
      const haystack = `${scenario.id} ${scenario.dataset} ${
        scenario.input_text
      } ${scenario.ground_truth_text || ""}`.toLowerCase();
      return haystack.includes(query);
    })
    .sort((a, b) => a.id.localeCompare(b.id));
}

function renderScenarioList() {
  scenarioListEl.innerHTML = "";
  const filtered = getFilteredScenarios();
  listMetaEl.textContent = `${filtered.length} of ${state.data.scenarios.length}`;

  filtered.forEach((scenario) => {
    const item = document.createElement("div");
    item.className = "list-item";
    if (scenario.id === state.selectedId) {
      item.classList.add("active");
    }

    const idEl = document.createElement("div");
    idEl.className = "list-id";
    idEl.textContent = scenario.id;

    const snippetEl = document.createElement("div");
    snippetEl.className = "list-snippet";
    snippetEl.textContent = truncate(scenario.input_text, 140);

    const tagsEl = document.createElement("div");
    tagsEl.className = "list-tags";

    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = scenario.dataset;
    tagsEl.appendChild(tag);

    item.appendChild(idEl);
    item.appendChild(snippetEl);
    item.appendChild(tagsEl);

    item.addEventListener("click", () => {
      state.selectedId = scenario.id;
      renderScenarioList();
      renderPanels();
    });

    scenarioListEl.appendChild(item);
  });
}

function detectLang(text) {
  const trimmed = (text || "").trim();
  if (
    (trimmed.startsWith("{") && trimmed.endsWith("}")) ||
    (trimmed.startsWith("[") && trimmed.endsWith("]"))
  ) {
    try {
      JSON.parse(trimmed);
      return "json";
    } catch {}
  }
  return "markdown";
}

function highlightCode(code, text) {
  code.className = `language-${detectLang(text)}`;
  hljs.highlightElement(code);
}

function buildTurnCard(turn) {
  const card = document.createElement("div");
  card.className = "turn";

  if (turn.role) {
    const role = document.createElement("div");
    role.className = "turn-role";
    role.textContent = turn.role;
    card.appendChild(role);
  }

  const pre = document.createElement("pre");
  pre.className = "turn-content";
  const code = document.createElement("code");
  code.textContent = turn.content || "";
  pre.appendChild(code);
  highlightCode(code, turn.content);
  card.appendChild(pre);

  return card;
}

function buildPreBlock(text) {
  const pre = document.createElement("pre");
  pre.className = "panel-pre";
  const code = document.createElement("code");
  code.textContent = text || "";
  pre.appendChild(code);
  highlightCode(code, text);
  return pre;
}

function buildEmptyState(text) {
  const empty = document.createElement("div");
  empty.className = "empty-state";
  empty.textContent = text;
  return empty;
}

function getSelectedScenario() {
  return state.data?.scenarios.find((item) => item.id === state.selectedId) || null;
}

function findResultForScenario(scenario) {
  const results = scenario.results || [];
  if (!results.length) {
    return { result: null, note: "" };
  }

  if (state.selectedModel) {
    const match = results.find((r) => r.model_name === state.selectedModel);
    if (match) {
      return { result: match, note: "" };
    }
    return {
      result: results[0],
      note: `Selected model not available — showing ${results[0].model_name}`,
    };
  }

  return { result: results[0], note: "" };
}

function stringify(value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "string") return value;
  return JSON.stringify(value, null, 2);
}

function formatRubric(template, scenario, modelOutput) {
  if (!template) return "";
  const vars = scenario.meta?.rubric_vars || {};
  return template.replace(/\{([a-zA-Z0-9_]+)\}/g, (match, key) => {
    if (key === "model_output") {
      return modelOutput || "";
    }
    if (key in vars) {
      return stringify(vars[key]);
    }
    return match;
  });
}

function renderPanels() {
  contextBodyEl.innerHTML = "";
  rubricBodyEl.innerHTML = "";
  responseBodyEl.innerHTML = "";
  judgeBodyEl.innerHTML = "";
  contextMetaEl.textContent = "";
  rubricMetaEl.textContent = "";
  responseMetaEl.textContent = "";
  judgeMetaEl.textContent = "";

  const scenario = getSelectedScenario();
  if (!scenario) {
    const empty = buildEmptyState("Select a scenario from the left.");
    contextBodyEl.appendChild(empty.cloneNode(true));
    rubricBodyEl.appendChild(empty.cloneNode(true));
    responseBodyEl.appendChild(empty.cloneNode(true));
    judgeBodyEl.appendChild(empty);
    return;
  }

  contextMetaEl.textContent = `${scenario.dataset} · ${scenario.source} (line ${scenario.line})`;
  if (scenario.input_turns.length) {
    scenario.input_turns.forEach((turn) => {
      contextBodyEl.appendChild(buildTurnCard(turn));
    });
  } else {
    contextBodyEl.appendChild(buildEmptyState("No context available."));
  }

  const { result, note } = findResultForScenario(scenario);
  let modelOutput = "";
  if (result && result.submission) {
    modelOutput = result.submission;
  } else if (scenario.ground_truth_text) {
    modelOutput = scenario.ground_truth_text;
  }

  const rubricTemplate = state.data?.rubric?.content || "";
  if (rubricTemplate) {
    rubricMetaEl.textContent = state.data?.rubric?.path
      ? `Source: ${state.data.rubric.path}`
      : "Rubric prompt";
    rubricBodyEl.appendChild(
      buildPreBlock(formatRubric(rubricTemplate, scenario, modelOutput))
    );
  } else if (scenario.ground_truth_text) {
    rubricMetaEl.textContent = "Ground truth (no rubric file)";
    rubricBodyEl.appendChild(buildPreBlock(scenario.ground_truth_text));
  } else {
    rubricMetaEl.textContent = "Rubric unavailable";
    rubricBodyEl.appendChild(buildEmptyState("No rubric text found."));
  }

  if (result) {
    const score =
      typeof result.score === "number"
        ? ` · Score ${(result.score * 100).toFixed(0)}%`
        : "";
    responseMetaEl.textContent = `${result.model_name}${score}`;
    if (note) {
      responseMetaEl.textContent += ` · ${note}`;
    }
    responseBodyEl.appendChild(buildPreBlock(result.submission || ""));

    // Judge panel
    if (result.rationale) {
      const scoreBadge = document.createElement("span");
      scoreBadge.className = "score-badge";
      if (typeof result.score === "number") {
        scoreBadge.textContent = `${(result.score * 100).toFixed(0)}%`;
        if (result.score >= 0.7) scoreBadge.classList.add("score-green");
        else if (result.score >= 0.4) scoreBadge.classList.add("score-yellow");
        else scoreBadge.classList.add("score-red");
      }
      judgeMetaEl.textContent = result.model_name;
      if (scoreBadge.textContent) {
        judgeMetaEl.textContent = "";
        const metaWrapper = document.createDocumentFragment();
        const modelText = document.createTextNode(`${result.model_name} `);
        metaWrapper.appendChild(modelText);
        metaWrapper.appendChild(scoreBadge);
        judgeMetaEl.appendChild(metaWrapper);
      }
      judgeBodyEl.appendChild(buildPreBlock(result.rationale));
    } else {
      judgeMetaEl.textContent = "No rationale";
      judgeBodyEl.appendChild(buildEmptyState("No judge rationale available."));
    }
  } else if (scenario.ground_truth_text) {
    responseMetaEl.textContent = "Ground truth (no model response)";
    responseBodyEl.appendChild(buildPreBlock(scenario.ground_truth_text));
    judgeMetaEl.textContent = "No results";
    judgeBodyEl.appendChild(buildEmptyState("No judge rationale available."));
  } else {
    responseMetaEl.textContent = "No model response";
    responseBodyEl.appendChild(buildEmptyState("No response available."));
    judgeMetaEl.textContent = "No results";
    judgeBodyEl.appendChild(buildEmptyState("No judge rationale available."));
  }
}

const handleSearch = debounce((value) => {
  state.search = value;
  const filtered = getFilteredScenarios();
  if (!filtered.find((item) => item.id === state.selectedId)) {
    state.selectedId = filtered[0]?.id || null;
  }
  renderScenarioList();
  renderPanels();
}, 150);

searchInputEl.addEventListener("input", (event) => {
  handleSearch(event.target.value || "");
});

modelSelectEl.addEventListener("change", (event) => {
  state.selectedModel = event.target.value || null;
  renderPanels();
});

const panelViews = {
  left: { context: { body: contextBodyEl, meta: contextMetaEl }, rubric: { body: rubricBodyEl, meta: rubricMetaEl } },
  right: { response: { body: responseBodyEl, meta: responseMetaEl }, judge: { body: judgeBodyEl, meta: judgeMetaEl } },
};

document.querySelectorAll(".panel-tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    const panel = tab.dataset.panel;
    const view = tab.dataset.view;
    // Toggle tab active state
    document.querySelectorAll(`.panel-tab[data-panel="${panel}"]`).forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    // Toggle body/meta visibility
    for (const [key, els] of Object.entries(panelViews[panel])) {
      const show = key === view;
      els.body.hidden = !show;
      els.meta.hidden = !show;
    }
  });
});

loadIndex();
