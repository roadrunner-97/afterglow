#include "MetadataReader.h"

#include <QImage>
#include <QTemporaryDir>
#include <QTest>

#include <libexif/exif-data.h>

#include <cstdlib>
#include <cstring>

namespace {
ExifEntry *addEntry(ExifData *data, ExifIfd ifd, ExifTag tag, ExifFormat format, unsigned int components) {
    ExifEntry *entry  = exif_entry_new();
    entry->tag        = tag;
    entry->format     = format;
    entry->components = components;
    entry->size       = exif_format_get_size(format) * components;
    entry->data       = static_cast<unsigned char *>(std::calloc(entry->size, 1));
    exif_content_add_entry(data->ifd[ifd], entry);
    exif_entry_unref(entry);
    return entry;
}

void addText(ExifData *data, ExifIfd ifd, ExifTag tag, const char *value) {
    const unsigned int length = static_cast<unsigned int>(std::strlen(value) + 1U);
    ExifEntry         *entry  = addEntry(data, ifd, tag, EXIF_FORMAT_ASCII, length);
    std::memcpy(entry->data, value, entry->size);
}

void addShort(ExifData *data, ExifIfd ifd, ExifTag tag, unsigned short value) {
    ExifEntry *entry = addEntry(data, ifd, tag, EXIF_FORMAT_SHORT, 1);
    exif_set_short(entry->data, exif_data_get_byte_order(data), value);
}

void addLong(ExifData *data, ExifIfd ifd, ExifTag tag, unsigned int value) {
    ExifEntry *entry = addEntry(data, ifd, tag, EXIF_FORMAT_LONG, 1);
    exif_set_long(entry->data, exif_data_get_byte_order(data), value);
}

void addByte(ExifData *data, ExifIfd ifd, ExifTag tag, unsigned char value) {
    ExifEntry *entry = addEntry(data, ifd, tag, EXIF_FORMAT_BYTE, 1);
    entry->data[0]   = value;
}

void addRational(ExifData *data, ExifIfd ifd, ExifTag tag, unsigned int numerator, unsigned int denominator) {
    ExifEntry *entry = addEntry(data, ifd, tag, EXIF_FORMAT_RATIONAL, 1);
    exif_set_rational(entry->data, exif_data_get_byte_order(data), {numerator, denominator});
}

void addSRational(ExifData *data, ExifIfd ifd, ExifTag tag, int numerator, int denominator) {
    ExifEntry *entry = addEntry(data, ifd, tag, EXIF_FORMAT_SRATIONAL, 1);
    exif_set_srational(entry->data, exif_data_get_byte_order(data), {numerator, denominator});
}

void addCoordinates(ExifData *data, ExifTag tag, unsigned int degrees, unsigned int minutes, unsigned int seconds) {
    ExifEntry          *entry = addEntry(data, EXIF_IFD_GPS, tag, EXIF_FORMAT_RATIONAL, 3);
    const ExifByteOrder order = exif_data_get_byte_order(data);
    exif_set_rational(entry->data, order, {degrees, 1});
    exif_set_rational(entry->data + sizeof(ExifRational), order, {minutes, 1});
    exif_set_rational(entry->data + 2 * sizeof(ExifRational), order, {seconds, 1});
}

QString createJpegWithExif(const QTemporaryDir &directory, bool withDateOriginal) {
    const QString path = directory.filePath(withDateOriginal ? "rich.jpg" : "fallback.jpg");
    QImage        image(2, 3, QImage::Format_RGB32);
    image.fill(Qt::black);
    if (!image.save(path, "JPG")) return {};

    ExifData *data = exif_data_new();
    exif_data_set_option(data, EXIF_DATA_OPTION_FOLLOW_SPECIFICATION);
    exif_data_set_data_type(data, EXIF_DATA_TYPE_COMPRESSED);
    exif_data_set_byte_order(data, EXIF_BYTE_ORDER_INTEL);

    addText(data, EXIF_IFD_0, EXIF_TAG_MAKE, "Afterglow");
    addText(data, EXIF_IFD_0, EXIF_TAG_MODEL, "Test Camera");
    addText(data, EXIF_IFD_0, EXIF_TAG_SOFTWARE, "Afterglow Tests");
    addText(data, EXIF_IFD_0, EXIF_TAG_ARTIST, "Tester");
    addText(data, EXIF_IFD_0, EXIF_TAG_COPYRIGHT, "Copyright");
    addText(data, EXIF_IFD_0, EXIF_TAG_IMAGE_DESCRIPTION, "Fixture");
    addText(data, EXIF_IFD_0, EXIF_TAG_DATE_TIME, "2026:09:21 12:34:56");
    addText(data, EXIF_IFD_EXIF, static_cast<ExifTag>(0xa434), "Test Lens");
    addText(data, EXIF_IFD_EXIF, static_cast<ExifTag>(0xa431), "12345");
    if (withDateOriginal) addText(data, EXIF_IFD_EXIF, EXIF_TAG_DATE_TIME_ORIGINAL, "2025:01:02 03:04:05");
    addLong(data, EXIF_IFD_EXIF, EXIF_TAG_PIXEL_X_DIMENSION, 2);
    addByte(data, EXIF_IFD_EXIF, EXIF_TAG_PIXEL_Y_DIMENSION, 3);
    addShort(data, EXIF_IFD_EXIF, EXIF_TAG_ISO_SPEED_RATINGS, 400);
    addShort(data, EXIF_IFD_EXIF, EXIF_TAG_EXPOSURE_PROGRAM, 1);
    addShort(data, EXIF_IFD_EXIF, EXIF_TAG_METERING_MODE, 5);
    addShort(data, EXIF_IFD_EXIF, EXIF_TAG_FLASH, 0);
    addShort(data, EXIF_IFD_EXIF, static_cast<ExifTag>(0xa403), 1);
    addRational(data, EXIF_IFD_EXIF, EXIF_TAG_EXPOSURE_TIME, 1, 125);
    addRational(data, EXIF_IFD_EXIF, EXIF_TAG_FNUMBER, 28, 10);
    addShort(data, EXIF_IFD_EXIF, EXIF_TAG_FOCAL_LENGTH, 50);
    addSRational(data, EXIF_IFD_EXIF, EXIF_TAG_EXPOSURE_BIAS_VALUE, -1, 3);
    addText(data, EXIF_IFD_GPS, static_cast<ExifTag>(EXIF_TAG_GPS_LATITUDE_REF), "S");
    addText(data, EXIF_IFD_GPS, static_cast<ExifTag>(EXIF_TAG_GPS_LONGITUDE_REF), "W");
    addCoordinates(data, static_cast<ExifTag>(EXIF_TAG_GPS_LATITUDE), 51, 30, 0);
    addCoordinates(data, static_cast<ExifTag>(EXIF_TAG_GPS_LONGITUDE), 0, 7, 30);

    exif_data_fix(data);
    unsigned char *exifBytes = nullptr;
    unsigned int   exifSize  = 0;
    exif_data_save_data(data, &exifBytes, &exifSize);
    exif_data_unref(data);

    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) return {};
    const QByteArray jpeg = file.readAll();
    file.close();
    if (jpeg.size() < 2 || !file.open(QIODevice::WriteOnly | QIODevice::Truncate)) return {};
    file.write(jpeg.left(2));
    file.write(reinterpret_cast<const char *>(exifBytes), exifSize);
    file.write(jpeg.mid(2));
    file.close();
    std::free(exifBytes);
    return path;
}
} // namespace

class TestMetadataReader : public QObject {
    Q_OBJECT

private slots:
    void rejectsNullDestination() {
        QVERIFY(!MetadataReader::read("unused.jpg", nullptr));
    }

    void readsJpegExifFields() {
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        const QString path = createJpegWithExif(directory, true);
        QVERIFY(!path.isEmpty());

        ImageMetadata metadata;
        QVERIFY(MetadataReader::read(path, &metadata));
        QCOMPARE(metadata.cameraMake, QString("Afterglow"));
        QCOMPARE(metadata.cameraModel, QString("Test Camera"));
        QCOMPARE(metadata.lens, QString("Test Lens"));
        QCOMPARE(metadata.cameraSerial, QString("12345"));
        QCOMPARE(metadata.software, QString("Afterglow Tests"));
        QCOMPARE(metadata.artist, QString("Tester"));
        QVERIFY(metadata.copyright.startsWith(QString("Copyright")));
        QCOMPARE(metadata.description, QString("Fixture"));
        QCOMPARE(metadata.pixelSize, QSize(2, 0));
        QCOMPARE(metadata.captureTime, QDateTime(QDate(2025, 1, 2), QTime(3, 4, 5)));
        QCOMPARE(metadata.isoSpeed, 400.0f);
        QCOMPARE(metadata.shutterSec, 1.0f / 125.0f);
        QCOMPARE(metadata.aperture, 2.8f);
        QCOMPARE(metadata.focalLenMm, 0.0f);
        QCOMPARE(metadata.exposureBiasEv, -1.0f / 3.0f);
        QVERIFY(!metadata.exposureProgram.isEmpty());
        QVERIFY(!metadata.meteringMode.isEmpty());
        QVERIFY(!metadata.flash.isEmpty());
        QVERIFY(!metadata.whiteBalance.isEmpty());
        QCOMPARE(metadata.location, QString("-51.50000, -0.12500"));
    }

    void fallsBackToFileDateWhenOriginalDateIsAbsent() {
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        const QString path = createJpegWithExif(directory, false);
        QVERIFY(!path.isEmpty());

        ImageMetadata metadata;
        QVERIFY(MetadataReader::read(path, &metadata));
        QCOMPARE(metadata.captureTime, QDateTime(QDate(2026, 9, 21), QTime(12, 34, 56)));
    }
};

QTEST_MAIN(TestMetadataReader)
#include "test_metadata_reader.moc"
