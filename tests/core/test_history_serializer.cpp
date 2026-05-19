#include <QTest>
#include <QTemporaryFile>
#include "HistorySerializer.h"
#include "SettingsImporter.h"

static SettingsImporter::EffectSettings makeShadow(
    const QString& id, bool enabled, QMap<QString, QVariant> params)
{
    SettingsImporter::EffectSettings e;
    e.id = id; e.name = id; e.enabled = enabled; e.parameters = std::move(params);
    return e;
}

static UndoHistory::Entry makeEntry(
    const QString& effectId, QMap<QString, QPair<QVariant,QVariant>> params,
    std::optional<std::pair<bool,bool>> enabled = std::nullopt)
{
    UndoHistory::Entry e;
    e.effectId = effectId;
    e.enabled  = enabled;
    for (auto it = params.cbegin(); it != params.cend(); ++it)
        e.params.insert(it.key(), {it.value().first, it.value().second});
    return e;
}

class TestHistorySerializer : public QObject {
    Q_OBJECT

private slots:
    void roundTripEmpty() {
        const QString yaml = HistorySerializer::toYaml({}, 0, {});
        HistorySerializer::HistoryData data;
        QVERIFY(HistorySerializer::fromYaml(yaml, &data));
        QCOMPARE(data.cursor, 0);
        QVERIFY(data.entries.isEmpty());
        QVERIFY(data.shadow.isEmpty());
    }

    void roundTripCursor() {
        QVector<UndoHistory::Entry> entries;
        entries << makeEntry("brightness", {{"value", {QVariant(0), QVariant(10)}}});
        QVector<SettingsImporter::EffectSettings> shadow;
        shadow << makeShadow("brightness", true, {{"value", 10}});

        HistorySerializer::HistoryData data;
        QVERIFY(HistorySerializer::fromYaml(
            HistorySerializer::toYaml(entries, 1, shadow), &data));

        QCOMPARE(data.cursor, 1);
        QCOMPARE(data.entries.size(), 1);
        QCOMPARE(data.entries[0].effectId, QString("brightness"));
        QVERIFY(data.entries[0].params.contains("value"));
        QCOMPARE(data.entries[0].params["value"].from, QVariant(0));
        QCOMPARE(data.entries[0].params["value"].to,   QVariant(10));
    }

    void roundTripEnabledDelta() {
        QVector<UndoHistory::Entry> entries;
        entries << makeEntry("vignette", {}, std::make_optional(std::make_pair(false, true)));

        HistorySerializer::HistoryData data;
        QVERIFY(HistorySerializer::fromYaml(
            HistorySerializer::toYaml(entries, 1, {}), &data));

        QCOMPARE(data.entries.size(), 1);
        QVERIFY(data.entries[0].enabled.has_value());
        QVERIFY(!data.entries[0].enabled->first);
        QVERIFY(data.entries[0].enabled->second);
    }

    void roundTripShadow() {
        QVector<SettingsImporter::EffectSettings> shadow;
        shadow << makeShadow("brightness", true,  {{"value", 18}});
        shadow << makeShadow("vignette",   false, {{"amount", 35}, {"feather", 50}});

        HistorySerializer::HistoryData data;
        QVERIFY(HistorySerializer::fromYaml(
            HistorySerializer::toYaml({}, 0, shadow), &data));

        QCOMPARE(data.shadow.size(), 2);
        QCOMPARE(data.shadow[0].id, QString("brightness"));
        QVERIFY(data.shadow[0].enabled);
        QCOMPARE(data.shadow[0].parameters.value("value"), QVariant(18));
        QCOMPARE(data.shadow[1].id, QString("vignette"));
        QVERIFY(!data.shadow[1].enabled);
        QCOMPARE(data.shadow[1].parameters.value("amount"), QVariant(35));
        QCOMPARE(data.shadow[1].parameters.value("feather"), QVariant(50));
    }

    void roundTripMultipleEntries() {
        QVector<UndoHistory::Entry> entries;
        entries << makeEntry("brightness", {{"value", {QVariant(0),  QVariant(12)}}});
        entries << makeEntry("brightness", {{"value", {QVariant(12), QVariant(18)}}});

        HistorySerializer::HistoryData data;
        QVERIFY(HistorySerializer::fromYaml(
            HistorySerializer::toYaml(entries, 2, {}), &data));

        QCOMPARE(data.entries.size(), 2);
        QCOMPARE(data.cursor, 2);
        QCOMPARE(data.entries[0].params["value"].from, QVariant(0));
        QCOMPARE(data.entries[0].params["value"].to,   QVariant(12));
        QCOMPARE(data.entries[1].params["value"].from, QVariant(12));
        QCOMPARE(data.entries[1].params["value"].to,   QVariant(18));
    }

    void roundTripDoubleValues() {
        QVector<UndoHistory::Entry> entries;
        entries << makeEntry("saturation", {{"amount", {QVariant(0.5), QVariant(1.5)}}});

        HistorySerializer::HistoryData data;
        QVERIFY(HistorySerializer::fromYaml(
            HistorySerializer::toYaml(entries, 1, {}), &data));

        QCOMPARE(data.entries[0].params["amount"].from.toDouble(), 0.5);
        QCOMPARE(data.entries[0].params["amount"].to.toDouble(),   1.5);
    }

    void roundTripBoolParam() {
        QVector<UndoHistory::Entry> entries;
        entries << makeEntry("effect", {{"flag", {QVariant(false), QVariant(true)}}});

        HistorySerializer::HistoryData data;
        QVERIFY(HistorySerializer::fromYaml(
            HistorySerializer::toYaml(entries, 1, {}), &data));

        QCOMPARE(data.entries[0].params["flag"].from.toBool(), false);
        QCOMPARE(data.entries[0].params["flag"].to.toBool(),   true);
    }

    void roundTripBothEnabledAndParams() {
        QVector<UndoHistory::Entry> entries;
        entries << makeEntry("vignette",
            {{"amount", {QVariant(0), QVariant(35)}}},
            std::make_optional(std::make_pair(false, true)));

        HistorySerializer::HistoryData data;
        QVERIFY(HistorySerializer::fromYaml(
            HistorySerializer::toYaml(entries, 1, {}), &data));

        QVERIFY(data.entries[0].enabled.has_value());
        QVERIFY(!data.entries[0].enabled->first);
        QVERIFY(data.entries[0].enabled->second);
        QCOMPARE(data.entries[0].params["amount"].from, QVariant(0));
        QCOMPARE(data.entries[0].params["amount"].to,   QVariant(35));
    }

    void fileRoundTrip() {
        QVector<UndoHistory::Entry> entries;
        entries << makeEntry("brightness", {{"value", {QVariant(5), QVariant(15)}}});
        QVector<SettingsImporter::EffectSettings> shadow;
        shadow << makeShadow("brightness", true, {{"value", 15}});

        QTemporaryFile tmp;
        QVERIFY(tmp.open());
        const QString path = tmp.fileName();
        tmp.close();

        QVERIFY(HistorySerializer::writeYaml(path, entries, 1, shadow));

        HistorySerializer::HistoryData data;
        QVERIFY(HistorySerializer::readYaml(path, &data));
        QCOMPARE(data.cursor, 1);
        QCOMPARE(data.entries.size(), 1);
        QCOMPARE(data.shadow.size(), 1);
        QCOMPARE(data.entries[0].effectId, QString("brightness"));
        QCOMPARE(data.shadow[0].id,        QString("brightness"));
    }

    void commentsAndBlankLinesIgnored() {
        const QString yaml =
            "# Afterglow undo history\n"
            "\n"
            "cursor: 3\n"
            "# another comment\n"
            "\n"
            "shadow:\n"
            "entries:\n";
        HistorySerializer::HistoryData data;
        QVERIFY(HistorySerializer::fromYaml(yaml, &data));
        QCOMPARE(data.cursor, 3);
        QVERIFY(data.shadow.isEmpty());
        QVERIFY(data.entries.isEmpty());
    }

    void writeFailsForBadPath() {
        QString error;
        const bool ok = HistorySerializer::writeYaml(
            "/nonexistent_dir/history.yml", {}, 0, {}, &error);
        QVERIFY(!ok);
        QVERIFY(!error.isEmpty());
    }

    void roundTripEmptyShadowParameters() {
        // Shadow entry with no parameters hits the `parameters: {}` branch (line 165).
        QVector<SettingsImporter::EffectSettings> shadow;
        shadow << makeShadow("brightness", true, {});

        HistorySerializer::HistoryData data;
        QVERIFY(HistorySerializer::fromYaml(
            HistorySerializer::toYaml({}, 0, shadow), &data));

        QCOMPARE(data.shadow.size(), 1);
        QVERIFY(data.shadow[0].parameters.isEmpty());
    }

    void roundTripEscapedCharactersInId() {
        // Exercise quoteString escape branches (lines 19-23) and the
        // corresponding unquoteStr decode branches (lines 68-71).
        QVector<SettingsImporter::EffectSettings> shadow;
        shadow << makeShadow("id\"with\\quotes\nand\rnewlines\ttabs",
                              true, {});

        HistorySerializer::HistoryData data;
        QVERIFY(HistorySerializer::fromYaml(
            HistorySerializer::toYaml({}, 0, shadow), &data));

        QCOMPARE(data.shadow.size(), 1);
        QCOMPARE(data.shadow[0].id,
                 QString("id\"with\\quotes\nand\rnewlines\ttabs"));
    }

    void roundTripStringVariantParam() {
        // A QVariant of type QString hits the default branch in formatScalar
        // (lines 50-51) and is written as a quoted string.
        QVector<UndoHistory::Entry> entries;
        UndoHistory::Entry e;
        e.effectId = "effect";
        e.params["label"] = {QVariant(QString("before")), QVariant(QString("after"))};
        entries.append(e);

        HistorySerializer::HistoryData data;
        QVERIFY(HistorySerializer::fromYaml(
            HistorySerializer::toYaml(entries, 1, {}), &data));

        QCOMPARE(data.entries[0].params["label"].from.toString(), QString("before"));
        QCOMPARE(data.entries[0].params["label"].to.toString(),   QString("after"));
    }

    void roundTripLargeIntParam() {
        // Value larger than INT_MAX exercises the large-longlong branch in
        // parseScalarV (line 104).
        const long long big = static_cast<long long>(INT_MAX) + 1;
        QVector<UndoHistory::Entry> entries;
        UndoHistory::Entry e;
        e.effectId = "e";
        e.params["v"] = {QVariant(static_cast<long long>(0)), QVariant(big)};
        entries.append(e);

        HistorySerializer::HistoryData data;
        QVERIFY(HistorySerializer::fromYaml(
            HistorySerializer::toYaml(entries, 1, {}), &data));

        QCOMPARE(data.entries[0].params["v"].to.toLongLong(), big);
    }

    void readFailsForMissingFile() {
        HistorySerializer::HistoryData data;
        QString error;
        QVERIFY(!HistorySerializer::readYaml("/no/such/file.yml", &data, &error));
        QVERIFY(!error.isEmpty());
    }
};

QTEST_APPLESS_MAIN(TestHistorySerializer)
#include "test_history_serializer.moc"
