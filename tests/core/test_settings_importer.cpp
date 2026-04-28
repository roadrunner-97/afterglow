#include <QTest>
#include <QSignalSpy>
#include <QTemporaryFile>
#include <memory>
#include "EffectManager.h"
#include "SettingsExporter.h"
#include "SettingsImporter.h"

namespace {

// A FakeEffect that records the parameter map applied to it, so tests can
// assert applyToManager() pushed the expected values.  applyParameters is
// non-virtual on the base only as a convenience, so this override exercises
// the dispatch path.
class FakeEffect : public PhotoEditorEffect {
    Q_OBJECT
public:
    FakeEffect(QString name, QMap<QString, QVariant> params = {})
        : m_name(std::move(name)), m_params(std::move(params)) {}
    QString getName()        const override { return m_name; }
    QString getDescription() const override { return ""; }
    QString getVersion()     const override { return "1.0"; }
    bool    initialize()           override { return true; }
    QImage processImage(const QImage& img, const QMap<QString,QVariant>&) override { return img; }
    QMap<QString, QVariant> getParameters() const override { return m_params; }
    void applyParameters(const QMap<QString, QVariant>& p) override {
        m_lastApplied = p;
        m_params = p;
        ++m_applyCalls;
        // Mirror real effect behaviour: applying parameters notifies
        // listeners that they changed.  SettingsImporter is expected to
        // silence this notification during a bulk apply.
        emit parametersChanged();
    }

    QMap<QString, QVariant> lastApplied() const { return m_lastApplied; }
    int applyCalls() const { return m_applyCalls; }

private:
    QString                 m_name;
    QMap<QString, QVariant> m_params;
    QMap<QString, QVariant> m_lastApplied;
    int                     m_applyCalls = 0;
};

} // namespace

class TestSettingsImporter : public QObject {
    Q_OBJECT

private slots:
    void parses_emptyDocumentYieldsNoEffects() {
        SettingsImporter::Settings s;
        QVERIFY(SettingsImporter::fromYaml("", &s, nullptr));
        QVERIFY(s.image.isEmpty());
        QVERIFY(s.effects.isEmpty());
    }

    void parses_imageAndScalarTypes() {
        const QString yaml = QStringLiteral(
            "# leading comment\n"
            "image: \"/tmp/x.jpg\"\n"
            "effects:\n"
            "  - name: \"Brightness\"\n"
            "    enabled: true\n"
            "    parameters:\n"
            "      brightness: 5\n"
            "      contrast: -7\n"
            "  - name: \"FilmGrain\"\n"
            "    enabled: false\n"
            "    parameters:\n"
            "      lumWeight: false\n"
            "      amount: 1.5\n"
        );

        SettingsImporter::Settings s;
        QVERIFY(SettingsImporter::fromYaml(yaml, &s, nullptr));
        QCOMPARE(s.image, QString("/tmp/x.jpg"));
        QCOMPARE(s.effects.size(), 2);

        QCOMPARE(s.effects[0].name, QString("Brightness"));
        QVERIFY(s.effects[0].enabled);
        QCOMPARE(s.effects[0].parameters.value("brightness").toInt(), 5);
        QCOMPARE(s.effects[0].parameters.value("contrast").toInt(),  -7);

        QCOMPARE(s.effects[1].name, QString("FilmGrain"));
        QVERIFY(!s.effects[1].enabled);
        QCOMPARE(s.effects[1].parameters.value("lumWeight").toBool(), false);
        QCOMPARE(s.effects[1].parameters.value("amount").toDouble(), 1.5);
    }

    void parses_emptyParametersFlowMap() {
        const QString yaml = QStringLiteral(
            "effects:\n"
            "  - name: \"X\"\n"
            "    enabled: true\n"
            "    parameters: {}\n");
        SettingsImporter::Settings s;
        QVERIFY(SettingsImporter::fromYaml(yaml, &s, nullptr));
        QCOMPARE(s.effects.size(), 1);
        QVERIFY(s.effects[0].parameters.isEmpty());
    }

    void unquotes_escapeSequences() {
        const QString yaml = QStringLiteral(
            "image: \"a\\nb\\tc\\\"d\\\\e\\x01f\"\n");
        SettingsImporter::Settings s;
        QVERIFY(SettingsImporter::fromYaml(yaml, &s, nullptr));
        const QString expected = QStringLiteral("a\nb\tc\"d\\e\x01""f");
        QCOMPARE(s.image, expected);
    }

    void unquotes_carriageReturnAndUnknownEscape() {
        // \r decodes to CR; an unknown escape like \q falls through the
        // switch's default branch and yields the literal escape char.
        const QString yaml = QStringLiteral("image: \"x\\ry\\qz\"\n");
        SettingsImporter::Settings s;
        QVERIFY(SettingsImporter::fromYaml(yaml, &s, nullptr));
        QCOMPARE(s.image, QStringLiteral("x\ry""qz"));
    }

    void unquotes_malformedHexLeavesXThenChars() {
        // \xZZ — hex parse fails, so the 'x' is appended and Z's continue
        // through the loop as ordinary characters.
        const QString yaml = QStringLiteral("image: \"\\xZZ\"\n");
        SettingsImporter::Settings s;
        QVERIFY(SettingsImporter::fromYaml(yaml, &s, nullptr));
        QCOMPARE(s.image, QStringLiteral("xZZ"));
    }

    void unquotes_unterminatedQuoteReturnedAsIs() {
        // Token starts with " but doesn't end with " — unquote bails out
        // and returns the raw token.
        const QString yaml = QStringLiteral("image: \"unterminated\n");
        SettingsImporter::Settings s;
        QVERIFY(SettingsImporter::fromYaml(yaml, &s, nullptr));
        QCOMPARE(s.image, QStringLiteral("\"unterminated"));
    }

    void parses_unquotedFallbackKeepsString() {
        // bare token that isn't bool/int/double — kept as the raw string.
        const QString yaml = QStringLiteral(
            "effects:\n"
            "  - name: \"X\"\n"
            "    enabled: true\n"
            "    parameters:\n"
            "      mode: hello\n");
        SettingsImporter::Settings s;
        QVERIFY(SettingsImporter::fromYaml(yaml, &s, nullptr));
        QCOMPARE(s.effects[0].parameters.value("mode").toString(),
                 QString("hello"));
    }

    void parses_largeIntegerPromotesToLongLong() {
        // Value beyond INT_MAX must round-trip without losing precision.
        const QString yaml = QStringLiteral(
            "effects:\n"
            "  - name: \"X\"\n"
            "    enabled: true\n"
            "    parameters:\n"
            "      big: 99999999999\n");
        SettingsImporter::Settings s;
        QVERIFY(SettingsImporter::fromYaml(yaml, &s, nullptr));
        QCOMPARE(s.effects[0].parameters.value("big").toLongLong(),
                 99999999999LL);
    }

    void roundTrip_exporterToImporter() {
        // Build a manager, export, parse — values should match what we wrote.
        EffectManager mgr;
        QMap<QString, QVariant> p;
        p["i"] = 42;
        p["d"] = 1.25;
        p["b"] = true;
        mgr.addEffect(std::make_unique<FakeEffect>("Test", p), /*enabled=*/false);

        const QString yaml = SettingsExporter::toYaml(mgr, "/tmp/round.jpg");
        SettingsImporter::Settings parsed;
        QVERIFY(SettingsImporter::fromYaml(yaml, &parsed, nullptr));

        QCOMPARE(parsed.image, QString("/tmp/round.jpg"));
        QCOMPARE(parsed.effects.size(), 1);
        // Exporter now writes id (snake-cased getName); name is left
        // empty unless an older sidecar provided it.
        QCOMPARE(parsed.effects[0].id, QString("test"));
        QVERIFY(parsed.effects[0].name.isEmpty());
        QVERIFY(!parsed.effects[0].enabled);
        QCOMPARE(parsed.effects[0].parameters.value("i").toInt(), 42);
        QCOMPARE(parsed.effects[0].parameters.value("d").toDouble(), 1.25);
        QCOMPARE(parsed.effects[0].parameters.value("b").toBool(), true);
    }

    void applyToManager_pushesEnabledAndParameters() {
        EffectManager mgr;
        auto owned = std::make_unique<FakeEffect>("Brightness");
        auto* fake = owned.get();
        mgr.addEffect(std::move(owned), /*enabled=*/true);

        SettingsImporter::Settings s;
        SettingsImporter::EffectSettings entry;
        entry.name = "Brightness";
        entry.enabled = false;
        entry.parameters["brightness"] = 10;
        entry.parameters["contrast"]   = 20;
        s.effects.append(entry);

        SettingsImporter::applyToManager(s, mgr);

        QVERIFY(!mgr.entries()[0].enabled);
        QCOMPARE(fake->applyCalls(), 1);
        QCOMPARE(fake->lastApplied().value("brightness").toInt(), 10);
        QCOMPARE(fake->lastApplied().value("contrast").toInt(),   20);
    }

    void applyToManager_skipsUnknownEffects() {
        EffectManager mgr;
        auto owned = std::make_unique<FakeEffect>("Brightness");
        auto* fake = owned.get();
        mgr.addEffect(std::move(owned));

        SettingsImporter::Settings s;
        SettingsImporter::EffectSettings entry;
        entry.name = "DoesNotExist";
        entry.parameters["x"] = 1;
        s.effects.append(entry);

        SettingsImporter::applyToManager(s, mgr);
        QCOMPARE(fake->applyCalls(), 0);
    }

    void applyToManager_doesNotEmitParametersChanged() {
        // Each applyParameters() would normally emit parametersChanged() and
        // queue a pipeline reprocess; the importer is expected to silence
        // those so the caller can fire one definitive reprocess at the end.
        EffectManager mgr;
        auto ownedA = std::make_unique<FakeEffect>("A");
        auto ownedB = std::make_unique<FakeEffect>("B");
        auto* a = ownedA.get();
        auto* b = ownedB.get();
        mgr.addEffect(std::move(ownedA));
        mgr.addEffect(std::move(ownedB));

        QSignalSpy spyA(a, &PhotoEditorEffect::parametersChanged);
        QSignalSpy spyB(b, &PhotoEditorEffect::parametersChanged);

        SettingsImporter::Settings s;
        SettingsImporter::EffectSettings ea; ea.name = "A"; ea.parameters["x"] = 1;
        SettingsImporter::EffectSettings eb; eb.name = "B"; eb.parameters["y"] = 2;
        s.effects.append(ea);
        s.effects.append(eb);

        SettingsImporter::applyToManager(s, mgr);

        QCOMPARE(a->applyCalls(), 1);
        QCOMPARE(b->applyCalls(), 1);
        QCOMPARE(spyA.count(), 0);
        QCOMPARE(spyB.count(), 0);
    }

    void applyToManager_fallsBackToNameForLegacySidecars() {
        // Sidecars written before the id migration use `- name:` keyed
        // entries; applyToManager must still match them by display name.
        EffectManager mgr;
        auto owned = std::make_unique<FakeEffect>("Brightness");
        auto* fake = owned.get();
        mgr.addEffect(std::move(owned));

        SettingsImporter::Settings s;
        SettingsImporter::EffectSettings entry;
        entry.name = "Brightness";  // legacy: id is empty
        entry.parameters["x"] = 7;
        s.effects.append(entry);

        SettingsImporter::applyToManager(s, mgr);
        QCOMPARE(fake->applyCalls(), 1);
        QCOMPARE(fake->lastApplied().value("x").toInt(), 7);
    }

    void applyToManager_leavesUntouchedEffectsAlone() {
        EffectManager mgr;
        auto ownedA = std::make_unique<FakeEffect>("A");
        auto ownedB = std::make_unique<FakeEffect>("B");
        auto* a = ownedA.get();
        auto* b = ownedB.get();
        mgr.addEffect(std::move(ownedA));
        mgr.addEffect(std::move(ownedB));

        SettingsImporter::Settings s;
        SettingsImporter::EffectSettings entry;
        entry.name = "A";
        s.effects.append(entry);

        SettingsImporter::applyToManager(s, mgr);
        QCOMPARE(a->applyCalls(), 1);
        QCOMPARE(b->applyCalls(), 0);
    }

    void readYaml_loadsFile() {
        EffectManager mgr;
        mgr.addEffect(std::make_unique<FakeEffect>("X", QMap<QString, QVariant>{{"k", 7}}));
        const QString yaml = SettingsExporter::toYaml(mgr, "/tmp/img.jpg");

        QTemporaryFile tmp;
        QVERIFY(tmp.open());
        tmp.write(yaml.toUtf8());
        tmp.close();

        SettingsImporter::Settings parsed;
        QString error;
        QVERIFY(SettingsImporter::readYaml(tmp.fileName(), &parsed, &error));
        QVERIFY(error.isEmpty());
        QCOMPARE(parsed.effects.size(), 1);
        QCOMPARE(parsed.effects[0].parameters.value("k").toInt(), 7);
    }

    void fromYaml_rejectsTabsInLeadingWhitespace() {
        // Tabs would silently confuse the indent dispatch.  The parser now
        // refuses up-front with a line-numbered error.
        const QString yaml = QStringLiteral("effects:\n\t- name: \"X\"\n");
        SettingsImporter::Settings s;
        QString error;
        QVERIFY(!SettingsImporter::fromYaml(yaml, &s, &error));
        QVERIFY(error.contains("tabs"));
        QVERIFY(error.contains("line 2"));
    }

    void fromYaml_clearsErrorOnSuccess() {
        QString error = "stale";
        SettingsImporter::Settings s;
        QVERIFY(SettingsImporter::fromYaml("", &s, &error));
        QVERIFY(error.isEmpty());
    }

    void readYaml_returnsFalseOnMissingFile() {
        SettingsImporter::Settings parsed;
        QString error;
        QVERIFY(!SettingsImporter::readYaml("/nonexistent_xyz/__nope__.yml",
                                            &parsed, &error));
        QVERIFY(!error.isEmpty());
    }
};

QTEST_GUILESS_MAIN(TestSettingsImporter)
#include "test_settings_importer.moc"
