#pragma once

#include <QImage>
#include <functional>

// ============================================================================
// Synthetic QImage factories for effect unit tests
// ============================================================================

// Solid-color image in Format_RGB32.
inline QImage makeSolid(int w, int h, int r, int g, int b) {
    QImage img(w, h, QImage::Format_RGB32);
    img.fill(qRgb(r, g, b));
    return img;
}

// Solid-color image in Format_RGBX64 (16-bit per channel).
inline QImage makeSolid16bit(int w, int h, int r, int g, int b) {
    QImage img(w, h, QImage::Format_RGBX64);
    img.fill(QColor(r, g, b));
    return img;
}

// Horizontal gradient: black (left) → white (right), grey values only.
inline QImage makeGradient(int w, int h) {
    QImage img(w, h, QImage::Format_RGB32);
    for (int y = 0; y < h; ++y) {
        auto* row = reinterpret_cast<QRgb*>(img.scanLine(y));
        for (int x = 0; x < w; ++x) {
            int v = (w > 1) ? (x * 255 / (w - 1)) : 0;
            row[x] = qRgb(v, v, v);
        }
    }
    return img;
}

// Checkerboard of black and white cells.
inline QImage makeCheckerboard(int w, int h, int cellSize = 8) {
    QImage img(w, h, QImage::Format_RGB32);
    for (int y = 0; y < h; ++y) {
        auto* row = reinterpret_cast<QRgb*>(img.scanLine(y));
        for (int x = 0; x < w; ++x)
            row[x] = ((x / cellSize + y / cellSize) % 2) ? qRgb(255, 255, 255)
                                                          : qRgb(0,   0,   0);
    }
    return img;
}

// Left/right split: left half (rL,gL,bL), right half (rR,gR,bR).
inline QImage makeSplit(int w, int h,
                        int rL, int gL, int bL,
                        int rR, int gR, int bR) {
    QImage img(w, h, QImage::Format_RGB32);
    for (int y = 0; y < h; ++y) {
        auto* row = reinterpret_cast<QRgb*>(img.scanLine(y));
        for (int x = 0; x < w; ++x)
            row[x] = (x < w / 2) ? qRgb(rL, gL, bL) : qRgb(rR, gR, bR);
    }
    return img;
}

// ============================================================================
// Pixel accessors
// ============================================================================

inline int pixelR(const QImage& img, int x, int y) {
    return qRed(reinterpret_cast<const QRgb*>(img.constScanLine(y))[x]);
}
inline int pixelG(const QImage& img, int x, int y) {
    return qGreen(reinterpret_cast<const QRgb*>(img.constScanLine(y))[x]);
}
inline int pixelB(const QImage& img, int x, int y) {
    return qBlue(reinterpret_cast<const QRgb*>(img.constScanLine(y))[x]);
}

// ============================================================================
// Predicate helpers
// ============================================================================

// Returns true iff every pixel satisfies pred.
inline bool allPixels(const QImage& img, std::function<bool(QRgb)> pred) {
    for (int y = 0; y < img.height(); ++y) {
        const auto* row = reinterpret_cast<const QRgb*>(img.constScanLine(y));
        for (int x = 0; x < img.width(); ++x)
            if (!pred(row[x])) return false;
    }
    return true;
}
