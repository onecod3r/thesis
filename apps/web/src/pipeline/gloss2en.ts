/**
 * ASL gloss -> English, rule engine v1: a line-by-line port of
 * `sb.rescore.gloss2en.convert` (the reverse of `rules_v2`). Python is
 * authoritative; `test/gloss2en.test.ts` requires identical output on every
 * corpus sentence and on thousands of recognized (often ungrammatical) ones.
 *
 * It is the **offline baseline**: deterministic, instant, and weak on noun lists,
 * clause structure and aspect (`sign-to-speech-demo-analysis.md` §3). It also
 * turns any gloss sequence into fluent-looking English, so the UI always shows
 * the glosses next to it.
 */

import type { Lexicon } from "../../../shared-ts/src/contracts.ts";

const SURFACE: Record<string, string> = {
  thankyou: "thank you", callonphone: "call", glasswindow: "window",
  frenchfries: "French fries", icecream: "ice cream", haveto: "have to",
  TV: "TV", shhh: "shh", mom: "Mom", dad: "Dad", grandma: "Grandma",
  grandpa: "Grandpa", owie: "owie", potty: "potty",
};
const NAMES = new Set(["mom", "dad", "grandma", "grandpa"]);
const MASS = new Set(["water", "milk", "food", "cereal", "pizza", "icecream", "gum", "grass", "garbage",
  "chocolate", "rain", "snow", "TV", "home", "fish", "hair", "outside", "time"]);
const PLURAL = new Set(["jeans", "pajamas", "scissors", "stairs", "nuts", "frenchfries", "feet", "underwear", "lips"]);
const PLACES_IN = new Set(["bedroom", "room", "closet", "drawer", "refrigerator", "dryer", "car", "boat", "airplane",
  "pool", "store", "farm", "backyard", "tree", "bed", "garbage", "water", "cloud", "hair"]);
const PLACES_ON = new Set(["table", "chair", "stairs", "flower", "horse"]);
const NOT_SPLIT = new Set(["hair", "water", "cloud", "horse", "tree"]);
const INTERJ = new Set(["hello", "bye", "yes", "no", "please", "thankyou", "shhh"]);
const TIME_FRONT = new Set(["yesterday", "tomorrow", "now"]);
const TIME_BACK: Record<string, string> = { morning: "in the morning", night: "at night", later: "later", now: "now" };
const FUTURE = new Set(["will", "tomorrow", "later"]);
const PAST = new Set(["yesterday"]);
const WH = new Set(["who", "where", "why"]);
const CONJ = new Set(["because", "if"]);

const VERBS: Record<string, [string, string, string]> = {
  bath: ["takes a bath", "took a bath", "take a bath"],
  shower: ["takes a shower", "took a shower", "take a shower"],
  nap: ["takes a nap", "took a nap", "take a nap"],
  wake: ["wakes up", "woke up", "wake up"],
  callonphone: ["calls", "called", "call"],
  blow: ["blows", "blew", "blow"], clean: ["cleans", "cleaned", "clean"],
  close: ["closes", "closed", "close"], cry: ["cries", "cried", "cry"],
  cut: ["cuts", "cut", "cut"], dance: ["dances", "danced", "dance"],
  drink: ["drinks", "drank", "drink"], drop: ["drops", "dropped", "drop"],
  dry: ["dries", "dried", "dry"], fall: ["falls", "fell", "fall"],
  find: ["finds", "found", "find"], finish: ["finishes", "finished", "finish"],
  give: ["gives", "gave", "give"], go: ["goes to", "went to", "go to"],
  hate: ["hates", "hated", "hate"], have: ["has", "had", "have"],
  hear: ["hears", "heard", "hear"], hide: ["hides", "hid", "hide"],
  jump: ["jumps", "jumped", "jump"], kiss: ["kisses", "kissed", "kiss"],
  like: ["likes", "liked", "like"], listen: ["listens to", "listened to", "listen to"],
  look: ["looks at", "looked at", "look at"], make: ["makes", "made", "make"],
  open: ["opens", "opened", "open"], pretend: ["pretends", "pretended", "pretend"],
  read: ["reads", "read", "read"], ride: ["rides", "rode", "ride"],
  say: ["says", "said", "say"], see: ["sees", "saw", "see"],
  sleep: ["sleeps", "slept", "sleep"], smile: ["smiles at", "smiled at", "smile at"],
  stay: ["stays", "stayed", "stay"], talk: ["talks", "talked", "talk"],
  taste: ["tastes", "tasted", "taste"], think: ["thinks", "thought", "think"],
  touch: ["touches", "touched", "touch"], vacuum: ["vacuums", "vacuumed", "vacuum"],
  wait: ["waits for", "waited for", "wait for"],
};
const OBJ_PREP_VERBS = new Set(["listen", "look", "smile", "wait", "go"]);
const NO_TO = new Set(["home", "outside"]);
const TIME_PHRASE: Record<string, string> = { morning: "in the morning", night: "at night" };
const PRONOUNS: Record<string, [string, string, string, string, string]> = {
  minemy: ["I", "me", "my", "mine", "1s"],
  yourself: ["you", "you", "your", "yours", "2"],
  weus: ["we", "us", "our", "ours", "1p"],
  hesheit: ["they", "them", "their", "theirs", "3p"],
};
const QUANT_DET: Record<string, string> = { all: "all the", another: "another", any: "any", every: "every", many: "many" };
const LONE_DET: Record<string, string> = { "all the": "everything", every: "everyone", another: "another one" };

type Kind = "np" | "verb" | "adj" | "neg" | "modal" | "prep" | "time" | "wh" | "conj";
type Chunk = [Kind, string[]];
type Carry = [string, string] | null;

const has = <T,>(o: Record<string, T>, k: string): boolean => Object.prototype.hasOwnProperty.call(o, k);
const tags = (g: string, lex: Lexicon): string[] => (has(lex, g) ? lex[g] : ["noun"]);
const is = (g: string, tag: string, lex: Lexicon): boolean => tags(g, lex).includes(tag);
const surface = (g: string): string => (has(SURFACE, g) ? SURFACE[g] : g);
const verbForms = (v: string): [string, string, string] => (has(VERBS, v) ? VERBS[v] : [v + "s", v + "ed", v]);
const verbBase = (v: string): string => (has(VERBS, v) ? VERBS[v][2] : v);
const firstWord = (s: string): string => s.split(/\s+/)[0];
const joinTruthy = (ws: string[]): string => ws.filter((w) => w).join(" ");

function nounPhrase(toks: string[], lex: Lexicon, role: "subj" | "obj"): [string, string] {
  if (toks.length === 1 && has(PRONOUNS, toks[0])) {
    const [subj, obj, , , pn] = PRONOUNS[toks[0]];
    return [role === "subj" ? subj : obj, pn];
  }
  if (toks.length === 1 && (toks[0] === "that" || toks[0] === "there")) return [toks[0], "3s"];
  const words: string[] = [];
  let det: string | null = null;
  const head = toks[toks.length - 1];
  for (const t of toks) {
    if (has(PRONOUNS, t)) det = PRONOUNS[t][2];
    else if (is(t, "quant", lex)) det = has(QUANT_DET, t) ? QUANT_DET[t] : t;
    else if (t === "that") det = "that";
    else words.push(surface(t));
  }
  if (words.length === 0) {
    const d = det ?? "";
    return [has(LONE_DET, d) ? LONE_DET[d] : d, "3s"];
  }
  const plural = PLURAL.has(head) || det === "many" || det === "all the";
  if ((det === "many" || det === "all the") && !PLURAL.has(head) && !MASS.has(head) && words.length) {
    const w = words[words.length - 1];
    const before = w.length >= 2 ? w[w.length - 2] : "";
    words[words.length - 1] = w.endsWith("y") && !"aeiou".includes(before)
      ? w.slice(0, -1) + "ies"
      : w + (w.endsWith("s") || w.endsWith("sh") || w.endsWith("ch") ? "es" : "s");
  }
  if (det === null && !NAMES.has(head) && !MASS.has(head) && !PLURAL.has(head)) det = "the";
  if (det === null && ["food", "hair", "water", "milk", "gum", "TV"].includes(head) && role === "subj") det = "the";
  const text = det ? [det, ...words].join(" ") : words.join(" ");
  return [text, plural ? "3p" : "3s"];
}

function chunks(toks: string[], lex: Lexicon): Chunk[] {
  const out: Chunk[] = [];
  const npBuf: string[] = [];
  const flush = () => {
    if (npBuf.length) {
      out.push(["np", npBuf.slice()]);
      npBuf.length = 0;
    }
  };
  toks.forEach((t, i) => {
    const tg = tags(t, lex);
    const prev = out.length ? out[out.length - 1][0] : null;
    const nxt = i + 1 < toks.length ? toks[i + 1] : null;
    if (WH.has(t)) {
      flush();
      out.push(["wh", [t]]);
    } else if (t === "not" || (t === "no" && tg.includes("neg"))) {
      flush();
      out.push(["neg", [t]]);
    } else if (t === "can" || t === "haveto" || t === "will") {
      flush();
      out.push(["modal", [t]]);
    } else if (tg.includes("time") && t !== "first" && !((t === "before" || t === "after") && nxt)) {
      flush();
      out.push(["time", [t]]);
    } else if (tg.includes("prep") && t !== "outside") {
      flush();
      out.push(["prep", [t]]);
    } else if (tg.includes("verb") && (!tg.includes("noun")
        || ((prev === "np" || prev === "neg" || prev === "modal" || prev === null) && (npBuf.length > 0 || !!prev))
        || (npBuf.length === 0 && prev === null && nxt !== null))) {
      if (tg.includes("adj") && npBuf.length && nxt === null) {
        flush();
        out.push(["adj", [t]]);
        return;
      }
      flush();
      out.push(["verb", [t]]);
    } else if (tg.includes("adj") && toks.length === 1) {
      out.push(["adj", [t]]);
    } else if (tg.includes("adj") && (npBuf.length > 0 || prev === "np" || prev === "adj" || prev === "neg" || prev === "verb")
        && !(nxt !== null && is(nxt, "noun", lex) && !is(nxt, "adj", lex) && npBuf.length === 0)) {
      flush();
      out.push(["adj", [t]]);
    } else {
      npBuf.push(t);
    }
  });
  flush();
  return splitNps(out, lex);
}

function splitNps(cs: Chunk[], lex: Lexicon): Chunk[] {
  const out: Chunk[] = [];
  for (const [k, t] of cs) {
    if (k !== "np" || t.length < 2) {
      out.push([k, t]);
      continue;
    }
    if (t[0] === "that" || t[0] === "there" || (has(PRONOUNS, t[0]) && t.slice(1).some((x) => has(PRONOUNS, x)))) {
      out.push(["np", [t[0]]], ["np", t.slice(1)]);
      continue;
    }
    const last = t[t.length - 1], prev = t[t.length - 2];
    if ((PLACES_IN.has(last) || PLACES_ON.has(last)) && !NOT_SPLIT.has(last) && is(prev, "noun", lex)
        && !has(PRONOUNS, prev) && !is(prev, "quant", lex)) {
      out.push(["np", t.slice(0, -1)], ["prep", [PLACES_ON.has(last) ? "on" : "in"]], ["np", [last]]);
      continue;
    }
    out.push([k, t]);
  }
  return out;
}

function conjugate(verb: string, pn: string, tense: string, neg: boolean, modal: string | null, hasObj: boolean): string {
  let [third, past, base] = verbForms(verb);
  if (OBJ_PREP_VERBS.has(verb) && !hasObj) [third, past, base] = [firstWord(third), firstWord(past), firstWord(base)];
  const sg3 = pn === "3s";
  if (modal === "can") return `can${neg ? "not" : ""} ${base}`;
  if (modal === "haveto") {
    if (neg) return `${sg3 ? "does" : "do"} not have to ${base}`;
    return `${sg3 ? "has" : "have"} to ${base}`;
  }
  if (tense === "future") return `will ${neg ? "not " : ""}${base}`;
  if (tense === "past") return neg ? `did not ${base}` : past;
  if (neg) return `${sg3 ? "does" : "do"} not ${base}`;
  return sg3 ? third : base;
}

function be(pn: string, tense: string): string {
  if (tense === "past") return pn === "1s" || pn === "3s" ? "was" : "were";
  if (tense === "future") return "will be";
  return pn === "1s" ? "am" : pn === "3s" ? "is" : "are";
}

function clause(toks: string[], lex: Lexicon, tense: string, inherit: Carry): [string, boolean, Carry] {
  const times = toks.filter((t) => has(TIME_PHRASE, t));
  const rest = toks.filter((t) => !has(TIME_PHRASE, t));
  let [text, q, subj] = clauseCore(rest, lex, tense, inherit);
  for (const t of times) text = text ? `${text} ${TIME_PHRASE[t]}` : TIME_PHRASE[t];
  return [text, q, subj];
}

function clauseCore(toks: string[], lex: Lexicon, tenseIn: string, inherit: Carry): [string, boolean, Carry] {
  let tense = tenseIn;
  const weatherRest = toks.filter((t) => !["now", "later", "tomorrow", "yesterday"].includes(t));
  if (weatherRest.length === 1 && (weatherRest[0] === "rain" || weatherRest[0] === "snow")) {
    const w = toks.includes("rain") ? "rain" : "snow";
    const verb = tense === "past" ? `${w}ed` : tense === "future" ? `will ${w}` : `is ${w}ing`;
    const r = toks.filter((t) => !["rain", "snow", "yesterday"].includes(t)).join(" ");
    return [`it ${verb}` + (r ? ` ${r}` : ""), false, null];
  }
  let cs = chunks(toks, lex);
  const question = cs.some(([k]) => k === "wh");
  const whChunk = cs.find(([k]) => k === "wh");
  const wh = whChunk ? whChunk[1][0] : null;
  cs = cs.filter(([k]) => k !== "wh");
  const neg = cs.some(([k]) => k === "neg");
  const modalChunk = cs.find(([k, t]) => k === "modal" && t[0] !== "will");
  const modal = modalChunk ? modalChunk[1][0] : null;
  if (cs.some(([k, t]) => k === "modal" && t[0] === "will")) tense = "future";
  const body = cs.filter(([k]) => k !== "neg" && k !== "modal");
  const verbs = body.filter(([k]) => k === "verb");
  if (verbs.length > 1 && verbs[verbs.length - 1][1][0] === "finish") {
    body.splice(body.indexOf(verbs[verbs.length - 1]), 1);
    tense = "past";
  } else if (verbs.length === 1 && verbs[0][1][0] === "finish" && tense === "present") {
    tense = "past";
  }

  let subj: string | null = null;
  let pn = "2";
  let rest = body;
  if (body.length && body[0][0] === "np") {
    rest = body.slice(1);
    [subj, pn] = nounPhrase(body[0][1], lex, "subj");
  } else if (inherit !== null && body.length && (body[0][0] === "adj" || body[0][0] === "verb")) {
    [subj, pn] = inherit;
  } else if (body.length && body.every(([k]) => k === "adj") && !question) {
    return ["be " + body.map(([, t]) => surface(t[0])).join(" and "), false, null];
  }
  let words: string[] = [];
  const verbI = rest.findIndex(([k]) => k === "verb");
  const adjI = rest.findIndex(([k]) => k === "adj");

  const tail = (items: Chunk[]): string[] => {
    const out: string[] = [];
    for (const [k, t] of items) {
      if (k === "np") out.push(nounPhrase(t, lex, "obj")[0]);
      else if (k === "prep") out.push(t[0]);
      else if (k === "adj") out.push(surface(t[0]));
      else if (k === "verb") out.push(verbBase(t[0]));
      else if (k === "time") out.push(has(TIME_BACK, t[0]) ? TIME_BACK[t[0]] : t[0]);
    }
    return out;
  };

  if (verbI >= 0 && (adjI < 0 || verbI < adjI || subj === null)) {
    const pre = rest.slice(0, verbI), v = rest[verbI][1][0], post = rest.slice(verbI + 1);
    const hasObj = post.some(([k]) => k === "np")
      && !(v === "go" && post.length > 0 && post[0][0] === "np" && NO_TO.has(post[0][1][post[0][1].length - 1]));
    if (subj === null && pre.length === 1 && pre[0][0] === "adj" && post.length && post[0][0] === "np") {
      [subj, pn] = [pre[0][1][0] + " " + nounPhrase(post[0][1], lex, "subj")[0], "3s"];
    }
    if (subj === null && !question) {
      let base = verbBase(v);
      if (OBJ_PREP_VERBS.has(v) && !hasObj) base = firstWord(base);
      words = [...(neg ? ["do not"] : []), base, ...tail(post)];
    } else if (question && wh !== null && wh !== "who") {
      let aux = tense === "past" ? "did" : tense === "future" ? "will" : pn === "3s" ? "does" : "do";
      if (modal === "can") aux = "can";
      words = [wh, aux, ...(subj ? [subj] : ["you"]), ...(neg ? ["not"] : []), verbBase(v), ...tail(post)];
      return [words.join(" "), true, null];
    } else {
      if (wh === "who") [subj, pn] = ["who", "3s"];
      words = [subj || "you", conjugate(v, pn, tense, neg, modal, hasObj), ...tail(pre), ...tail(post)];
    }
  } else if (adjI >= 0) {
    let pre = rest.slice(0, adjI);
    const post = rest.slice(adjI);
    if (subj === null) {
      [subj, pn] = pre.length === 0 ? ["it", "3s"] : nounPhrase(pre[0][1], lex, "subj");
      pre = pre.length ? pre.slice(1) : pre;
    }
    const adjs = post.filter(([k]) => k === "adj").map(([, t]) => surface(t[0]));
    const others = post.filter(([k]) => k !== "adj");
    const adjText = adjs.length === 1 ? adjs[0] : adjs.slice(0, -1).join(", ") + " and " + adjs[adjs.length - 1];
    words = [subj, be(pn, tense), ...(neg ? ["not"] : []), adjText];
    for (const [k, t] of others) {
      if (k === "np") {
        const head = t[t.length - 1];
        const prep = PLACES_ON.has(head) ? "on" : PLACES_IN.has(head) ? "in" : "with";
        words.push(prep, nounPhrase(t, lex, "obj")[0]);
      } else {
        words.push(...tail([[k, t]]));
      }
    }
    if (question && wh) {
      words = [wh, be(pn, tense), subj, ...words.slice(2)];
      return [words.join(" "), true, null];
    }
  } else if (subj !== null && rest.length) {
    const [k0, t0] = rest[0];
    if (question && wh) {
      if ((wh === "who" && subj === "that") || (subj && wh === "who" && toks[0] === "that")) {
        const np = nounPhrase(t0, lex, "obj")[0];
        return [`whose ${np.startsWith("the ") ? np.slice(4) : np} is that`, true, null];
      }
      return [[wh, be(pn, tense), subj, ...tail(rest)].join(" "), true, null];
    }
    if (k0 === "prep") {
      words = [subj, be(pn, tense), ...(neg ? ["not"] : []), ...tail(rest)];
    } else if (k0 === "np") {
      const head = t0[t0.length - 1];
      if (["that", "there", "they", "I", "you", "we"].includes(subj) || has(PRONOUNS, toks[0]) || toks[0] === "that") {
        words = [subj, be(pn, tense), ...(neg ? ["not"] : []), ...tail(rest)];
      } else {
        const prep = PLACES_ON.has(head) ? "on" : PLACES_IN.has(head) ? "in" : "and";
        words = prep === "and" ? [subj, "and", ...tail(rest)] : [subj, be(pn, tense), prep, ...tail(rest)];
      }
    } else {
      words = [subj, ...tail(rest)];
    }
  } else if (subj !== null) {
    if (question && wh) return [`${wh} ${be(pn, tense)} ${subj}`, true, null];
    words = [...(neg ? ["no"] : []), subj];
  } else {
    words = [...(neg ? ["not"] : []), ...tail(rest)];
    if (question && wh) return [[wh, ...words].join(" "), true, null];
  }
  if (question && wh === "who" && words.length && words[0] !== "who") words = ["who", ...words.slice(1)];
  let carry: Carry = null;
  if (subj && !["that", "there", "who"].includes(subj)) {
    carry = ["I", "you", "we", "they"].includes(subj) ? [subj, pn] : ["they", "3p"];
  }
  return [joinTruthy(words), question, carry];
}

/** One gloss sentence (GISLR labels, ASL order) -> one English sentence. */
export function glossToEnglish(glosses: readonly string[], lexicon: Lexicon): string {
  let toks = [...glosses];
  if (!toks.length) return "";
  const lead: string[] = [], trail: string[] = [];
  while (toks.length && INTERJ.has(toks[0]) && (toks.length === 1 || toks[0] !== "no" || INTERJ.has(toks[1]))) {
    lead.push(toks.shift()!);
  }
  let question = false;
  if (toks.length >= 2 && toks[toks.length - 2] === "yes" && toks[toks.length - 1] === "no") {
    toks = toks.slice(0, -2);
    question = true;
  }
  if (toks.length && toks[toks.length - 1] === "please") trail.push(toks.pop()!);
  let sentenceNo = false;
  if (toks.length > 1 && toks[toks.length - 1] === "no") {
    toks = toks.slice(0, -1);
    sentenceNo = true;
  }
  const leadConj = toks.length > 1 && CONJ.has(toks[0]) ? toks.shift()! : null;
  const tense = toks.some((t) => PAST.has(t)) ? "past" : toks.some((t) => FUTURE.has(t)) ? "future" : "present";
  const front = toks.filter((t) => TIME_FRONT.has(t) && toks.indexOf(t) === 0);
  toks = toks.filter((t) => !front.includes(t));
  let parts: string[][] = [[]];
  let joins: string[] = [];
  if (leadConj && toks.length > 1 && (toks[0] === "rain" || toks[0] === "snow")) {
    parts = [[toks[0]], []];
    joins = [","];
    toks = toks.slice(1);
  }
  for (const t of toks) {
    if (CONJ.has(t) && parts[parts.length - 1].length) {
      parts.push([]);
      joins.push(t);
    } else {
      parts[parts.length - 1].push(t);
    }
  }
  const texts: string[] = [];
  let carry: Carry = null;
  for (const p of parts) {
    if (p.length) {
      const [txt, q, c] = clause(p, lexicon, tense, carry);
      carry = c;
      texts.push(txt);
      question = question || q;
    }
  }
  let body = texts.length ? texts[0] : "";
  for (let i = 0; i < Math.min(joins.length, texts.length - 1); i++) {
    const j = joins[i], t = texts[i + 1];
    body += j === "," ? `${j} ${t}` : ` ${j} ${t}`;
  }
  if (sentenceNo) body = "no " + body;
  if (leadConj) body = `${leadConj} ${body}`;
  const pieces = lead.length ? [lead.map(surface).join(" ")] : [];
  const head = front.map(surface).join(" ");
  if (head) body = body ? `${head} ${body}` : head;
  if (body) pieces.push(body);
  let out = pieces.join(", ");
  if (trail.length) out += ", please";
  out = out.trim();
  if (!out) return "";
  return out[0].toUpperCase() + out.slice(1) + (question ? "?" : ".");
}
