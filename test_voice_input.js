/*
 * test_voice_input.js
 * -------------------
 * Tests for the microphone input in the AI drawer (#100 / issue #81).
 *
 * Why these matter: every way voice input can fail ends in the same place —
 * SpeechRecognition's onend — but only the success path had ever run onstart,
 * which is what puts the overlay on screen. A refused microphone therefore
 * left the button looking simply dead: no overlay, no message, nothing but a
 * console line, on that click and every one after it. Dismissing the Chrome
 * permission prompt once is all it takes, and it is exactly what a first-time
 * visitor does.
 *
 * The functions are pulled straight out of dashboard.html and run against a
 * stubbed SpeechRecognition and a three-element DOM, so the failure paths can
 * be exercised without a microphone or a browser.
 *
 * Run with:  node test_voice_input.js
 */

const fs = require('fs');
const path = require('path');
const vm = require('vm');

const html = fs.readFileSync(path.join(__dirname, 'frontend', 'dashboard.html'), 'utf8');
const START = '    let recognition = null;';
const END = '    function initAssistantChat(force = false)';
if (!html.includes(START) || !html.includes(END)) {
  console.error('Could not find the voice input block in dashboard.html — has it moved?');
  process.exit(1);
}
const source = html.slice(html.indexOf(START), html.indexOf(END));

let audioContextsBuilt = 0;

function makeEnv() {
  const els = {
    aiSpeechOverlay: { style: { display: 'none' } },
    aiSpeechOverlayText: { innerText: '' },
    aiInput: { value: '', focus() { } },
  };
  let instance = null;

  class FakeRecognition {
    constructor() { instance = this; this.running = false; }
    start() {
      if (this.running) throw new Error('InvalidStateError');
      this.running = true;
    }
    stop() { if (this.running) { this.running = false; this.onend && this.onend(); } }
  }

  class FakeAudioContext {
    constructor() {
      audioContextsBuilt++;
      // Chrome has historically refused to build more than a handful per
      // document, which is the failure a per-beep context walks into.
      if (audioContextsBuilt > 6) throw new Error('too many AudioContexts');
    }
    get state() { return 'running'; }
    get currentTime() { return 0; }
    get destination() { return {}; }
    resume() { }
    createOscillator() {
      return {
        connect() { },
        frequency: { setValueAtTime() { }, exponentialRampToValueAtTime() { } },
        start() { }, stop() { },
      };
    }
    createGain() {
      return { connect() { }, gain: { setValueAtTime() { }, exponentialRampToValueAtTime() { } } };
    }
  }

  const sandbox = {
    document: { getElementById: id => els[id] || null },
    window: {
      SpeechRecognition: FakeRecognition,
      webkitSpeechRecognition: FakeRecognition,
      AudioContext: FakeAudioContext,
    },
    // Errors on the failure paths are expected; the overlay is what is under
    // test, not the logging.
    console: { warn() { }, error() { }, log: console.log },
    setTimeout,
    alert() { },
  };
  const ctx = vm.createContext(sandbox);
  vm.runInContext(source + '\n;this.toggle = toggleSpeechRecognition;', ctx);

  return {
    els,
    toggle: () => ctx.toggle(),
    rec: () => instance,
    // The browser fires onend after every session, however it ended.
    end: () => { instance.running = false; instance.onend(); },
  };
}

let failures = 0;
function check(name, condition, detail) {
  console.log((condition ? 'PASS  ' : 'FAIL  ') + name
    + (!condition && detail ? ' — ' + detail : ''));
  if (!condition) failures++;
}

// --- A refused microphone: onerror then onend, onstart never runs ----------
{
  const t = makeEnv();
  t.toggle();
  t.rec().onerror({ error: 'not-allowed' });
  t.end();
  check('a refused microphone shows the overlay',
    t.els.aiSpeechOverlay.style.display === 'flex',
    'display is ' + t.els.aiSpeechOverlay.style.display);
  check('a refused microphone names the reason',
    /Microphone access is blocked/.test(t.els.aiSpeechOverlayText.innerText),
    JSON.stringify(t.els.aiSpeechOverlayText.innerText));
  t.toggle();
  check('the mic still responds on the next click', t.rec().running === true);
}

// --- No microphone attached -------------------------------------------------
{
  const t = makeEnv();
  t.toggle();
  t.rec().onerror({ error: 'audio-capture' });
  t.end();
  check('a missing microphone is named',
    /No microphone found/.test(t.els.aiSpeechOverlayText.innerText),
    JSON.stringify(t.els.aiSpeechOverlayText.innerText));
}

// --- The speech service is unreachable -------------------------------------
{
  const t = makeEnv();
  t.toggle();
  t.rec().onerror({ error: 'network' });
  t.end();
  check('an unreachable speech service is named',
    /speech service/.test(t.els.aiSpeechOverlayText.innerText),
    JSON.stringify(t.els.aiSpeechOverlayText.innerText));
}

// --- Silence keeps the wording it always had -------------------------------
{
  const t = makeEnv();
  t.toggle();
  t.rec().onstart();
  t.rec().onerror({ error: 'no-speech' });
  t.end();
  check("silence still says \"Didn't catch that\"",
    /Didn't catch that/.test(t.els.aiSpeechOverlayText.innerText),
    JSON.stringify(t.els.aiSpeechOverlayText.innerText));
}

// --- The success path is unchanged -----------------------------------------
{
  const t = makeEnv();
  t.toggle();
  t.rec().onstart();
  t.rec().onresult({
    resultIndex: 0,
    results: [Object.assign([{ transcript: 'plan a route' }], { isFinal: true })],
  });
  t.end();
  check('the transcript reaches the input',
    t.els.aiInput.value === 'plan a route', JSON.stringify(t.els.aiInput.value));
  check('the overlay closes on success',
    t.els.aiSpeechOverlay.style.display === 'none');
}

// --- A transcript is appended to what is already typed ---------------------
{
  const t = makeEnv();
  t.els.aiInput.value = 'compare';
  t.toggle();
  t.rec().onstart();
  t.rec().onresult({
    resultIndex: 0,
    results: [Object.assign([{ transcript: 'the two plans' }], { isFinal: true })],
  });
  t.end();
  check('a transcript is appended, not substituted',
    t.els.aiInput.value === 'compare the two plans', JSON.stringify(t.els.aiInput.value));
}

// --- A second click before onstart stops rather than starting again --------
{
  const t = makeEnv();
  t.toggle();
  t.toggle();
  check('a second click before onstart stops instead of restarting',
    t.rec().running === false);
}

// --- One AudioContext for the page, not one per beep -----------------------
{
  audioContextsBuilt = 0;
  const t = makeEnv();
  for (let i = 0; i < 8; i++) {
    t.toggle();
    t.rec().onstart();
    t.end();
  }
  check('eight voice rounds build one AudioContext, not sixteen',
    audioContextsBuilt === 1, 'built ' + audioContextsBuilt);
}

console.log();
if (failures) {
  console.log(failures + ' check(s) failed');
  process.exit(1);
}
console.log('All voice input checks passed.');
