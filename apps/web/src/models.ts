/**
 * Where the pages read their model bundles from: this site (`/assets/...`, the default) or,
 * with `?models=kaggle`, the public Kaggle Models copies pinned in `kaggle.models.json`.
 * Kaggle serves both hops of a download (kaggle.com -> signed storage.googleapis.com URL) with
 * CORS headers, so a plain `fetch` works from any origin.
 */

import pins from "../kaggle.models.json";

export type Bundle = keyof typeof pins.bundles;

export function kaggleUrl(bundle: Bundle): string {
  const b = pins.bundles[bundle];
  return `https://www.kaggle.com/api/v1/models/${pins.owner}/${b.model}/${b.framework}/${bundle}/${b.version}/download`;
}

/** Base URL, without a trailing slash, that `bundle`'s files are fetched from. */
export function bundleBase(bundle: Bundle, local: string, search: string = location.search): string {
  return new URLSearchParams(search).get("models") === "kaggle" ? kaggleUrl(bundle) : local;
}
