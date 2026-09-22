#include "LinearImageIO.h"

#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfFrameBuffer.h>
#include <OpenEXR/ImfHeader.h>
#include <OpenEXR/ImfInputFile.h>
#include <OpenEXR/ImfOutputFile.h>
#include <OpenEXR/ImfStringAttribute.h>
#include <Imath/ImathBox.h>

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QUuid>
#include <cstdio>
#include <exception>

namespace {

constexpr qsizetype FLOATS_PER_PIXEL = 4;

void setError(QString *error, const QString &message) {
    if (error) *error = message;
}

} // namespace

namespace LinearImageIO {

bool isExrPath(const QString &path) {
    return QFileInfo(path).suffix().compare(QStringLiteral("exr"), Qt::CaseInsensitive) == 0;
}

QImage readExr(const QString &path, QString *error) {
    try {
        const QByteArray   encoded = QFile::encodeName(path);
        Imf::InputFile     file(encoded.constData());
        const Imath::Box2i window = file.header().dataWindow();
        const int          width  = window.max.x - window.min.x + 1;
        const int          height = window.max.y - window.min.y + 1;
        // OpenEXR rejects inverted/empty data windows before InputFile opens.
        // GCOVR_EXCL_START
        if (width <= 0 || height <= 0) {
            setError(error, QStringLiteral("The EXR has invalid dimensions."));
            return {};
        }
        // GCOVR_EXCL_STOP
        const Imf::ChannelList &channels = file.header().channels();
        if (!channels.findChannel("R") || !channels.findChannel("G") || !channels.findChannel("B")) {
            setError(error, QStringLiteral("The EXR does not contain RGB channels."));
            return {};
        }

        QImage image(width, height, QImage::Format_RGBA32FPx4);
        // Allocation failure depends on system memory exhaustion.
        // GCOVR_EXCL_START
        if (image.isNull()) {
            setError(error, QStringLiteral("Could not allocate the EXR image."));
            return {};
        }
        // GCOVR_EXCL_STOP
        const size_t     xStride = FLOATS_PER_PIXEL * sizeof(float);
        const size_t     yStride = static_cast<size_t>(image.bytesPerLine());
        const qptrdiff   xOffset = static_cast<qptrdiff>(window.min.x) * static_cast<qptrdiff>(xStride);
        const qptrdiff   yOffset = static_cast<qptrdiff>(window.min.y) * static_cast<qptrdiff>(yStride);
        char            *base    = reinterpret_cast<char *>(image.bits()) - xOffset - yOffset;
        Imf::FrameBuffer frameBuffer;
        frameBuffer.insert("R", Imf::Slice(Imf::FLOAT, base, xStride, yStride));
        frameBuffer.insert("G", Imf::Slice(Imf::FLOAT, base + sizeof(float), xStride, yStride));
        frameBuffer.insert("B", Imf::Slice(Imf::FLOAT, base + 2 * sizeof(float), xStride, yStride));
        file.setFrameBuffer(frameBuffer);
        file.readPixels(window.min.y, window.max.y);

        for (int y = 0; y < height; ++y) {
            auto *row = reinterpret_cast<float *>(image.scanLine(y));
            for (int x = 0; x < width; ++x) row[FLOATS_PER_PIXEL * x + 3] = 1.0f;
        }
        image.setText(QStringLiteral("color_space"), QStringLiteral("linear"));
        return image;
    }
    // All predictable path/format failures are handled above; remaining
    // writer exceptions are OpenEXR/filesystem runtime failures.
    // GCOVR_EXCL_START
    catch (const std::exception &e) {
        setError(error, QStringLiteral("Could not read EXR %1: %2").arg(path, QString::fromLocal8Bit(e.what())));
        return {};
    }
}

bool writeExr(const QString &path, const QImage &image, QString *error) {
    if (image.isNull()) {
        setError(error, QStringLiteral("The linear image is empty."));
        return false;
    }
    const QImage linear =
        image.format() == QImage::Format_RGBA32FPx4 ? image : image.convertToFormat(QImage::Format_RGBA32FPx4);
    const QFileInfo destination(path);
    if (!QDir().mkpath(destination.absolutePath())) {
        setError(error, QStringLiteral("Could not create %1.").arg(destination.absolutePath()));
        return false;
    }
    const QString temporary = path + QStringLiteral(".tmp-") + QUuid::createUuid().toString(QUuid::Id128);
    try {
        {
            Imf::Header header(linear.width(), linear.height());
            header.channels().insert("R", Imf::Channel(Imf::FLOAT));
            header.channels().insert("G", Imf::Channel(Imf::FLOAT));
            header.channels().insert("B", Imf::Channel(Imf::FLOAT));
            header.insert("afterglowColorSpace", Imf::StringAttribute("scene-linear-sRGB"));

            const QByteArray encoded = QFile::encodeName(temporary);
            Imf::OutputFile  file(encoded.constData(), header);
            const size_t     xStride = FLOATS_PER_PIXEL * sizeof(float);
            const size_t     yStride = static_cast<size_t>(linear.bytesPerLine());
            // OutputFile only reads the pixels, but Slice requires a mutable pointer.
            // constBits() preserves QImage sharing and avoids copying the full image.
            char            *base = const_cast<char *>(reinterpret_cast<const char *>(linear.constBits()));
            Imf::FrameBuffer frameBuffer;
            frameBuffer.insert("R", Imf::Slice(Imf::FLOAT, base, xStride, yStride));
            frameBuffer.insert("G", Imf::Slice(Imf::FLOAT, base + sizeof(float), xStride, yStride));
            frameBuffer.insert("B", Imf::Slice(Imf::FLOAT, base + 2 * sizeof(float), xStride, yStride));
            file.setFrameBuffer(frameBuffer);
            file.writePixels(linear.height());
        }
        const QByteArray sourceName = QFile::encodeName(temporary);
        const QByteArray targetName = QFile::encodeName(path);
        if (::rename(sourceName.constData(), targetName.constData()) != 0) {
            QFile::remove(temporary);
            setError(error, QStringLiteral("Could not replace %1.").arg(path));
            return false;
        }
        return true;
    } catch (const std::exception &e) {
        QFile::remove(temporary);
        setError(error, QStringLiteral("Could not write EXR %1: %2").arg(path, QString::fromLocal8Bit(e.what())));
        return false;
    }
    // GCOVR_EXCL_STOP
}

} // namespace LinearImageIO
