#include <QTest>
#include <QTemporaryFile>
#include <QFile>
#include "EffectManager.h"
#include "SettingsExporter.h"

namespace {

// Mock effect with a configurable parameter map and name.
class FakeEffect : public PhotoEditorEffect {
    Q_OBJECT
public:
    FakeEffect(QString name, QMap<QString, QVariant> params)
        : m_name(std::move(name)), m_params(std::move(params)) {}
    QString getName()        const override { return m_name; }
    QString getDescription() const override { return ""; }
    QString getVersion()     const override { return "1.0"; }
    bool    initialize()           override { return true; }
    QImage processImage(const QImage& img, const QMap<QString,QVariant>&) override { return img; }
    QMap<QString, QVariant> getParameters() const override { return m_params; }
private:
    QString                 m_name;
    QMap<QString, QVariant> m_params;
};

} // namespace

class TestSettingsExporter : public QObject {
    Q_OBJECT

private slots:
    void emptyManager_emitsEmptyEffectsList() {
        EffectManager mgr;
        const QString yaml = SettingsExporter::toYaml(mgr);
        QVERIFY(yaml.contains("effects:\n"));
        QVERIFY(!yaml.contains("- name:"));
    }

    void omitsImagePathWhenEmpty() {
        EffectManager mgr;
        const QString yaml = SettingsExporter::toYaml(mgr);
        QVERIFY(!yaml.contains("image:"));
    }

    void emitsImagePathWhenProvided() {
        EffectManager mgr;
        const QString yaml = SettingsExporter::toYaml(mgr, "/tmp/photo.jpg");
        QVERIFY(yaml.contains("image: \"/tmp/photo.jpg\""));
    }

    void serialisesScalarKinds() {
        EffectManager mgr;
        QMap<QString, QVariant> params;
        params["i"] = 42;
        params["d"] = 1.5;
        params["b"] = true;
        mgr.addEffect(new FakeEffect("Test", params));

        const QString yaml = SettingsExporter::toYaml(mgr);
        QVERIFY(yaml.contains("- name: \"Test\""));
        QVERIFY(yaml.contains("enabled: true"));
        QVERIFY(yaml.contains("i: 42"));
        QVERIFY(yaml.contains("d: 1.5"));
        QVERIFY(yaml.contains("b: true"));
    }

    void respectsDisabledEffects() {
        EffectManager mgr;
        mgr.addEffect(new FakeEffect("Off", {}), /*enabled=*/false);
        QVERIFY(SettingsExporter::toYaml(mgr).contains("enabled: false"));
    }

    void emptyParametersAreFlowMap() {
        EffectManager mgr;
        mgr.addEffect(new FakeEffect("NoParams", {}));
        QVERIFY(SettingsExporter::toYaml(mgr).contains("parameters: {}"));
    }

    void escapesQuotesAndBackslashesInPaths() {
        EffectManager mgr;
        const QString yaml = SettingsExporter::toYaml(mgr, "/has \"quote\" and \\back");
        QVERIFY(yaml.contains("\\\""));
        QVERIFY(yaml.contains("\\\\"));
    }

    void escapesNewlineCarriageTabAndControlChars() {
        EffectManager mgr;
        const QString path = QStringLiteral("a\nb\rc\td\x01");
        const QString yaml = SettingsExporter::toYaml(mgr, path);
        QVERIFY(yaml.contains("\\n"));
        QVERIFY(yaml.contains("\\r"));
        QVERIFY(yaml.contains("\\t"));
        QVERIFY(yaml.contains("\\x01"));
    }

    void serialisesStringValuedParameter() {
        // Exercises the QString fallback in formatScalar for non-numeric values.
        EffectManager mgr;
        QMap<QString, QVariant> params;
        params["mode"] = QString("auto");
        mgr.addEffect(new FakeEffect("Mode", params));
        QVERIFY(SettingsExporter::toYaml(mgr).contains("mode: \"auto\""));
    }

    void writeYaml_createsFile() {
        EffectManager mgr;
        mgr.addEffect(new FakeEffect("Brightness", {{"brightness", 5}}));

        QTemporaryFile tmp;
        tmp.setAutoRemove(true);
        QVERIFY(tmp.open());
        const QString path = tmp.fileName();
        tmp.close();

        QString error;
        QVERIFY(SettingsExporter::writeYaml(path, mgr, "/tmp/x.jpg", &error));
        QVERIFY(error.isEmpty());

        QFile f(path);
        QVERIFY(f.open(QIODevice::ReadOnly));
        const QString content = QString::fromUtf8(f.readAll());
        QVERIFY(content.contains("- name: \"Brightness\""));
        QVERIFY(content.contains("brightness: 5"));
        QVERIFY(content.contains("image: \"/tmp/x.jpg\""));
    }

    void writeYaml_returnsFalseOnUnwritablePath() {
        EffectManager mgr;
        QString error;
        // Path under a non-existent directory — open() must fail.
        const bool ok = SettingsExporter::writeYaml(
            "/nonexistent_dir_xyz/__nope__/out.yml", mgr, {}, &error);
        QVERIFY(!ok);
        QVERIFY(!error.isEmpty());
    }
};

QTEST_GUILESS_MAIN(TestSettingsExporter)
#include "test_settings_exporter.moc"
