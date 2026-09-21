#include "MetadataReader.h"
#include "RawLoader.h"

#ifdef HAVE_LIBEXIF
#include <libexif/exif-data.h>
#endif

#include <QDateTime>
#include <QLocale>
#include <cmath>

namespace {
#ifdef HAVE_LIBEXIF
QString entryText(ExifData *data, ExifIfd ifd, ExifTag tag) {
    ExifEntry *entry = exif_content_get_entry(data->ifd[ifd], tag);
    if (!entry) return {};
    char value[1024]{};
    exif_entry_get_value(entry, value, sizeof(value));
    return QString::fromUtf8(value).trimmed();
}

double rationalValue(ExifData *data, ExifIfd ifd, ExifTag tag) {
    ExifEntry *entry = exif_content_get_entry(data->ifd[ifd], tag);
    if (!entry || !entry->data || entry->components < 1) return 0.0;
    const ExifByteOrder order = exif_data_get_byte_order(data);
    if (entry->format == EXIF_FORMAT_RATIONAL) {
        const ExifRational r = exif_get_rational(entry->data, order);
        return r.denominator ? static_cast<double>(r.numerator) / r.denominator : 0.0;
    }
    if (entry->format == EXIF_FORMAT_SRATIONAL) {
        const ExifSRational r = exif_get_srational(entry->data, order);
        return r.denominator ? static_cast<double>(r.numerator) / r.denominator : 0.0;
    }
    return 0.0;
}

int integerValue(ExifData *data, ExifIfd ifd, ExifTag tag) {
    ExifEntry *entry = exif_content_get_entry(data->ifd[ifd], tag);
    if (!entry || !entry->data || entry->components < 1) return 0;
    const ExifByteOrder order = exif_data_get_byte_order(data);
    if (entry->format == EXIF_FORMAT_SHORT) return exif_get_short(entry->data, order);
    if (entry->format == EXIF_FORMAT_LONG) return static_cast<int>(exif_get_long(entry->data, order));
    return 0;
}

double gpsCoordinate(ExifData *data, ExifTag coordinateTag, ExifTag referenceTag) {
    ExifEntry *entry = exif_content_get_entry(data->ifd[EXIF_IFD_GPS], coordinateTag);
    if (!entry || !entry->data || entry->format != EXIF_FORMAT_RATIONAL || entry->components < 3) return 0.0;
    const ExifByteOrder order = exif_data_get_byte_order(data);
    double              parts[3]{};
    for (unsigned int i = 0; i < 3; ++i) {
        const ExifRational value = exif_get_rational(entry->data + i * sizeof(ExifRational), order);
        if (value.denominator) parts[i] = static_cast<double>(value.numerator) / value.denominator;
    }
    double        coordinate = parts[0] + parts[1] / 60.0 + parts[2] / 3600.0;
    const QString reference  = entryText(data, EXIF_IFD_GPS, referenceTag);
    if (reference.startsWith('S', Qt::CaseInsensitive) || reference.startsWith('W', Qt::CaseInsensitive))
        coordinate = -coordinate;
    return coordinate;
}

QDateTime exifDateTime(const QString &value) {
    return QDateTime::fromString(value.left(19), QStringLiteral("yyyy:MM:dd HH:mm:ss"));
}

QString cleanTag(const QString &value) {
    return value == QStringLiteral("Unknown") ? QString() : value;
}
#endif
} // namespace

bool MetadataReader::read(const QString &filePath, ImageMetadata *metadata) {
    if (!metadata) return false;
    *metadata = {};
    if (RawLoader::isRawFile(filePath)) return RawLoader::loadMetadata(filePath, metadata);

#ifdef HAVE_LIBEXIF
    ExifData *data = exif_data_new_from_file(filePath.toLocal8Bit().constData());
    if (!data) return false;

    metadata->cameraMake   = entryText(data, EXIF_IFD_0, EXIF_TAG_MAKE);
    metadata->cameraModel  = entryText(data, EXIF_IFD_0, EXIF_TAG_MODEL);
    metadata->lens         = entryText(data, EXIF_IFD_EXIF, static_cast<ExifTag>(0xa434)); // LensModel
    metadata->cameraSerial = entryText(data, EXIF_IFD_EXIF, static_cast<ExifTag>(0xa431));
    metadata->software     = entryText(data, EXIF_IFD_0, EXIF_TAG_SOFTWARE);
    metadata->artist       = entryText(data, EXIF_IFD_0, EXIF_TAG_ARTIST);
    metadata->copyright    = entryText(data, EXIF_IFD_0, EXIF_TAG_COPYRIGHT);
    metadata->description  = entryText(data, EXIF_IFD_0, EXIF_TAG_IMAGE_DESCRIPTION);
    const int pixelWidth   = integerValue(data, EXIF_IFD_EXIF, EXIF_TAG_PIXEL_X_DIMENSION);
    const int pixelHeight  = integerValue(data, EXIF_IFD_EXIF, EXIF_TAG_PIXEL_Y_DIMENSION);
    metadata->pixelSize    = QSize(pixelWidth, pixelHeight);
    metadata->captureTime  = exifDateTime(entryText(data, EXIF_IFD_EXIF, EXIF_TAG_DATE_TIME_ORIGINAL));
    if (!metadata->captureTime.isValid())
        metadata->captureTime = exifDateTime(entryText(data, EXIF_IFD_0, EXIF_TAG_DATE_TIME));

    metadata->shutterSec     = static_cast<float>(rationalValue(data, EXIF_IFD_EXIF, EXIF_TAG_EXPOSURE_TIME));
    metadata->aperture       = static_cast<float>(rationalValue(data, EXIF_IFD_EXIF, EXIF_TAG_FNUMBER));
    metadata->focalLenMm     = static_cast<float>(rationalValue(data, EXIF_IFD_EXIF, EXIF_TAG_FOCAL_LENGTH));
    metadata->exposureBiasEv = static_cast<float>(rationalValue(data, EXIF_IFD_EXIF, EXIF_TAG_EXPOSURE_BIAS_VALUE));

    const QString iso   = entryText(data, EXIF_IFD_EXIF, EXIF_TAG_ISO_SPEED_RATINGS);
    bool          isoOk = false;
    metadata->isoSpeed  = iso.section(' ', 0, 0).toFloat(&isoOk);
    if (!isoOk) metadata->isoSpeed = 0.0f;

    metadata->exposureProgram = cleanTag(entryText(data, EXIF_IFD_EXIF, EXIF_TAG_EXPOSURE_PROGRAM));
    metadata->meteringMode    = cleanTag(entryText(data, EXIF_IFD_EXIF, EXIF_TAG_METERING_MODE));
    metadata->flash           = cleanTag(entryText(data, EXIF_IFD_EXIF, EXIF_TAG_FLASH));
    metadata->whiteBalance    = cleanTag(entryText(data, EXIF_IFD_EXIF, static_cast<ExifTag>(0xa403)));

    const double latitude  = gpsCoordinate(data, static_cast<ExifTag>(EXIF_TAG_GPS_LATITUDE),
                                           static_cast<ExifTag>(EXIF_TAG_GPS_LATITUDE_REF));
    const double longitude = gpsCoordinate(data, static_cast<ExifTag>(EXIF_TAG_GPS_LONGITUDE),
                                           static_cast<ExifTag>(EXIF_TAG_GPS_LONGITUDE_REF));
    if (std::abs(latitude) > 1e-9 || std::abs(longitude) > 1e-9)
        metadata->location =
            QString::number(latitude, 'f', 5) + QStringLiteral(", ") + QString::number(longitude, 'f', 5);

    exif_data_unref(data);
    return true;
#else
    Q_UNUSED(filePath)
    return false;
#endif
}
