/**
 * Stage 4: the browser's own voices (`speechSynthesis`). The default on the
 * Workers Free plan (sign-to-speech-downstream.md §4): OS voices were fully
 * intelligible to Whisper (WER 0–0.3%) in about 0.2 s, at no cost.
 */

export function voices(): SpeechSynthesisVoice[] {
  if (!("speechSynthesis" in window)) return [];
  return speechSynthesis.getVoices().filter((v) => v.lang.toLowerCase().startsWith("en"));
}

export function onVoicesChanged(cb: () => void): void {
  if ("speechSynthesis" in window) speechSynthesis.addEventListener("voiceschanged", cb);
}

export function speak(text: string, voiceName?: string): Promise<void> {
  if (!("speechSynthesis" in window) || !text) return Promise.resolve();
  return new Promise((resolve) => {
    const u = new SpeechSynthesisUtterance(text);
    const v = voices().find((x) => x.name === voiceName);
    if (v) u.voice = v;
    u.lang = v?.lang ?? "en-US";
    u.onend = () => resolve();
    u.onerror = () => resolve();
    speechSynthesis.speak(u);
  });
}
