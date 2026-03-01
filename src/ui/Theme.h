#ifndef THEME_H
#define THEME_H

// Central palette for the 70s-inspired light-mode UI theme.
// Reference these instead of scattering hex literals across stylesheet strings.

namespace Theme {

// ── Background tones ─────────────────────────────────────────────────────────
constexpr const char* BG_MAIN         = "#F0EDE5";   // toolbar / central widget
constexpr const char* BG_RIGHT_PANEL  = "#EDEADE";   // right panel
constexpr const char* BG_EFFECT_PANEL = "#F4F1EA";   // per-effect card
constexpr const char* BG_SPINBOX      = "#F8F5F0";   // spinbox background

// ── Text ─────────────────────────────────────────────────────────────────────
constexpr const char* TEXT_PRIMARY    = "#2C2018";   // dark warm-brown primary text
constexpr const char* TEXT_SECONDARY  = "#6E5E46";   // muted secondary / labels

// ── Borders ──────────────────────────────────────────────────────────────────
constexpr const char* BORDER          = "#CCC5B5";   // standard border
constexpr const char* BORDER_PANEL    = "#CBBFAE";   // panel separator

// ── Accent colours ───────────────────────────────────────────────────────────
constexpr const char* ACCENT_STEEL    = "#5B6EA8";   // steel blue — slider fill/handle
constexpr const char* ACCENT_AMBER    = "#C0802C";   // warm amber — hover / checked

// ── Collapse button ──────────────────────────────────────────────────────────
constexpr const char* COLLAPSE_BG     = "#D0C8B8";
constexpr const char* COLLAPSE_HOVER  = "#BEB8A8";

// ── Checked/highlighted states ───────────────────────────────────────────────
constexpr const char* CHECKED_BG      = "#C0802C";
constexpr const char* CHECKED_TEXT    = "#F5F2EA";

} // namespace Theme

#endif // THEME_H
