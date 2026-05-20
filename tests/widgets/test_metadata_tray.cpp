#include <QApplication>
#include <QFormLayout>
#include <QLabel>
#include <QTest>
#include "MetadataTray.h"

static QLabel *findLabel(MetadataTray *t, const QString &text) {
    const auto labels = t->findChildren<QLabel *>();
    for (auto *l : labels) {
        if (l->text() == text) return l;
    }
    return nullptr;
}

// Collect all value QLabel texts into a list (skips the "Metadata" header and key labels).
// Key labels are short single-word strings; value labels either contain the em-dash or
// the data we set.  We identify value labels by checking the form layout's field column.
static QStringList valueTexts(MetadataTray *t) {
    const auto *fl = t->findChild<QFormLayout *>();
    if (!fl) return {};
    QStringList out;
    for (int i = 0; i < fl->rowCount(); ++i) {
        auto *item = fl->itemAt(i, QFormLayout::FieldRole);
        if (!item) continue;
        if (auto *lbl = qobject_cast<QLabel *>(item->widget())) out << lbl->text();
    }
    return out;
}

class TestMetadataTray : public QObject {
    Q_OBJECT

private slots:
    void constructsWithDashPlaceholders() {
        MetadataTray tray;
        const QStringList vals = valueTexts(&tray);
        QCOMPARE(vals.size(), 6);
        const QString dash = QString::fromUtf8("\xe2\x80\x94");
        for (const auto &v : vals) QCOMPARE(v, dash);
    }

    void setInfoPopulatesAllFields() {
        MetadataTray tray;
        MetadataTray::Info info;
        info.filename   = "DSC_0042.jpg";
        info.dimensions = "6000 x 4000";
        info.camera     = "Sony A7IV";
        info.lens       = "24-70mm f/2.8";
        info.exposure   = "ISO 400 1/250 s f/4.0";
        info.captured   = "2024-03-15 14:32";
        tray.setInfo(info);

        const QStringList vals = valueTexts(&tray);
        QCOMPARE(vals.size(), 6);
        QCOMPARE(vals[0], info.filename);
        QCOMPARE(vals[1], info.dimensions);
        QCOMPARE(vals[2], info.camera);
        QCOMPARE(vals[3], info.lens);
        QCOMPARE(vals[4], info.exposure);
        QCOMPARE(vals[5], info.captured);
    }

    void clearResetsToDashes() {
        MetadataTray tray;
        MetadataTray::Info info;
        info.filename = "photo.jpg";
        tray.setInfo(info);
        tray.clear();

        const QString     dash = QString::fromUtf8("\xe2\x80\x94");
        const QStringList vals = valueTexts(&tray);
        for (const auto &v : vals) QCOMPARE(v, dash);
    }

    void setInfoEmptyStringsShowDash() {
        MetadataTray    tray;
        MetadataTray::Info info; // all fields empty
        tray.setInfo(info);

        const QString     dash = QString::fromUtf8("\xe2\x80\x94");
        const QStringList vals = valueTexts(&tray);
        for (const auto &v : vals) QCOMPARE(v, dash);
    }

    void headerLabelExists() {
        MetadataTray tray;
        QVERIFY(findLabel(&tray, "Metadata") != nullptr);
    }
};

QTEST_MAIN(TestMetadataTray)
#include "test_metadata_tray.moc"
