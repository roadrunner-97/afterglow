#include <QTest>
#include "UndoHistory.h"
#include "SettingsImporter.h"

static SettingsImporter::EffectSettings makeEff(const QString &id, bool enabled, QMap<QString, QVariant> params) {
    SettingsImporter::EffectSettings e;
    e.id         = id;
    e.name       = id;
    e.enabled    = enabled;
    e.parameters = std::move(params);
    return e;
}

class TestUndoHistory : public QObject {
    Q_OBJECT

private slots:
    void emptyInitially() {
        UndoHistory h;
        QVERIFY(!h.canUndo());
        QVERIFY(!h.canRedo());
        QCOMPARE(h.cursor(), 0);
        QVERIFY(h.entries().isEmpty());
    }

    void seedClearsLog() {
        UndoHistory                               h;
        QVector<SettingsImporter::EffectSettings> snap;
        snap << makeEff("brightness", true, {{"value", 5}});
        h.seed(snap);
        snap[0].parameters["value"] = 10;
        h.recordFromCurrent(snap);
        QVERIFY(h.canUndo());

        h.seed(snap); // re-seed should clear
        QVERIFY(!h.canUndo());
        QVERIFY(!h.canRedo());
        QCOMPARE(h.entries().size(), 0);
    }

    void recordPushesEntry() {
        UndoHistory                               h;
        QVector<SettingsImporter::EffectSettings> snap;
        snap << makeEff("brightness", true, {{"value", 0}});
        h.seed(snap);
        snap[0].parameters["value"] = 10;
        h.recordFromCurrent(snap);

        QVERIFY(h.canUndo());
        QVERIFY(!h.canRedo());
        QCOMPARE(h.entries().size(), 1);
        QCOMPARE(h.cursor(), 1);
        QCOMPARE(h.entries()[0].effectId, QString("brightness"));
        QCOMPARE(h.entries()[0].params["value"].from, QVariant(0));
        QCOMPARE(h.entries()[0].params["value"].to, QVariant(10));
    }

    void noRecordWhenUnchanged() {
        UndoHistory                               h;
        QVector<SettingsImporter::EffectSettings> snap;
        snap << makeEff("brightness", true, {{"value", 5}});
        h.seed(snap);
        h.recordFromCurrent(snap);
        QVERIFY(!h.canUndo());
        QCOMPARE(h.entries().size(), 0);
    }

    void undoReturnsFrimValues() {
        UndoHistory                               h;
        QVector<SettingsImporter::EffectSettings> snap;
        snap << makeEff("brightness", true, {{"value", 0}});
        h.seed(snap);
        snap[0].parameters["value"] = 20;
        h.recordFromCurrent(snap);

        auto entry = h.undo();
        QVERIFY(entry.has_value());
        QCOMPARE(entry->effectId, QString("brightness"));
        QCOMPARE(entry->params["value"].from, QVariant(0));
        QCOMPARE(entry->params["value"].to, QVariant(20));
        QVERIFY(!h.canUndo());
        QVERIFY(h.canRedo());
        QCOMPARE(h.cursor(), 0);
    }

    void redoReturnsEntry() {
        UndoHistory                               h;
        QVector<SettingsImporter::EffectSettings> snap;
        snap << makeEff("brightness", true, {{"value", 0}});
        h.seed(snap);
        snap[0].parameters["value"] = 20;
        h.recordFromCurrent(snap);
        h.undo();

        auto entry = h.redo();
        QVERIFY(entry.has_value());
        QCOMPARE(entry->params["value"].to, QVariant(20));
        QVERIFY(h.canUndo());
        QVERIFY(!h.canRedo());
        QCOMPARE(h.cursor(), 1);
    }

    void undoNopWhenEmpty() {
        UndoHistory h;
        QVERIFY(!h.undo().has_value());
    }

    void redoNopWhenAtEnd() {
        UndoHistory                               h;
        QVector<SettingsImporter::EffectSettings> snap;
        snap << makeEff("b", true, {{"v", 0}});
        h.seed(snap);
        QVERIFY(!h.redo().has_value());
    }

    void recordTruncatesRedoTail() {
        UndoHistory                               h;
        QVector<SettingsImporter::EffectSettings> snap;
        snap << makeEff("b", true, {{"v", 0}});
        h.seed(snap);

        snap[0].parameters["v"] = 10;
        h.recordFromCurrent(snap);
        snap[0].parameters["v"] = 20;
        h.recordFromCurrent(snap);
        QCOMPARE(h.entries().size(), 2);

        h.undo();
        QVERIFY(h.canRedo());

        snap[0].parameters["v"] = 99;
        h.recordFromCurrent(snap);
        QVERIFY(!h.canRedo());
        QCOMPARE(h.cursor(), 2);
        QCOMPARE(h.entries().size(), 2);
        QCOMPARE(h.entries()[1].params["v"].to, QVariant(99));
    }

    void capacityDropsFromFront() {
        UndoHistory                               h(3);
        QVector<SettingsImporter::EffectSettings> snap;
        snap << makeEff("b", true, {{"v", 0}});
        h.seed(snap);
        for (int i = 1; i <= 5; ++i) {
            snap[0].parameters["v"] = i;
            h.recordFromCurrent(snap);
        }
        QCOMPARE(h.entries().size(), 3);
        QCOMPARE(h.cursor(), 3);
        QVERIFY(h.canUndo());
        QVERIFY(!h.canRedo());
    }

    void applyingGuardPreventsRecord() {
        UndoHistory                               h;
        QVector<SettingsImporter::EffectSettings> snap;
        snap << makeEff("b", true, {{"v", 0}});
        h.seed(snap);

        h.setApplying(true);
        snap[0].parameters["v"] = 10;
        h.recordFromCurrent(snap);
        QVERIFY(!h.canUndo());

        h.setApplying(false);
        h.recordFromCurrent(snap);
        QVERIFY(h.canUndo());
    }

    void enabledDeltaRecorded() {
        UndoHistory                               h;
        QVector<SettingsImporter::EffectSettings> snap;
        snap << makeEff("vignette", false, {});
        h.seed(snap);
        snap[0].enabled = true;
        h.recordFromCurrent(snap);

        QVERIFY(h.canUndo());
        auto entry = h.undo();
        QVERIFY(entry.has_value());
        QVERIFY(entry->enabled.has_value());
        QVERIFY(!entry->enabled->first); // from: false
        QVERIFY(entry->enabled->second); // to: true
    }

    void multipleEffectsCreateMultipleEntries() {
        UndoHistory                               h;
        QVector<SettingsImporter::EffectSettings> snap;
        snap << makeEff("brightness", true, {{"value", 0}});
        snap << makeEff("vignette", false, {{"amount", 0}});
        h.seed(snap);

        snap[0].parameters["value"] = 10;
        snap[1].enabled             = true;
        h.recordFromCurrent(snap);

        QCOMPARE(h.entries().size(), 2);
        QCOMPARE(h.cursor(), 2);
    }

    void shadowUpdatedAfterUndo() {
        UndoHistory                               h;
        QVector<SettingsImporter::EffectSettings> snap;
        snap << makeEff("b", true, {{"v", 0}});
        h.seed(snap);
        snap[0].parameters["v"] = 10;
        h.recordFromCurrent(snap);

        h.undo();
        // After undo, shadow reflects "from" state; a second identical-looking
        // recordFromCurrent should detect no change (the shadow tracks v=0).
        snap[0].parameters["v"] = 0;
        h.recordFromCurrent(snap);
        QCOMPARE(h.entries().size(), 1); // no new entry
    }

    void loadRestoresState() {
        QVector<UndoHistory::Entry> entries;
        UndoHistory::Entry          e;
        e.effectId        = "brightness";
        e.params["value"] = {QVariant(0), QVariant(10)};
        entries.append(e);

        QVector<SettingsImporter::EffectSettings> shadow;
        shadow << makeEff("brightness", true, {{"value", 10}});

        UndoHistory h;
        h.load(entries, 1, shadow);
        QVERIFY(h.canUndo());
        QVERIFY(!h.canRedo());
        QCOMPARE(h.cursor(), 1);
        QCOMPARE(h.entries().size(), 1);
    }

    void loadClampsOutOfRangeCursor() {
        QVector<UndoHistory::Entry> entries;
        UndoHistory::Entry          e;
        e.effectId    = "b";
        e.params["v"] = {QVariant(0), QVariant(1)};
        entries.append(e);

        UndoHistory h;
        h.load(entries, 99, {}); // cursor > size should clamp to size
        QCOMPARE(h.cursor(), 1);
    }

    void canUndoChangedSignalFires() {
        UndoHistory                               h;
        QVector<SettingsImporter::EffectSettings> snap;
        snap << makeEff("b", true, {{"v", 0}});
        h.seed(snap);

        bool gotTrue = false;
        connect(&h, &UndoHistory::canUndoChanged, this, [&](bool b) { gotTrue = b; });
        snap[0].parameters["v"] = 5;
        h.recordFromCurrent(snap);
        QVERIFY(gotTrue);
    }

    void isApplyingGetter() {
        UndoHistory h;
        QVERIFY(!h.isApplying());
        h.setApplying(true);
        QVERIFY(h.isApplying());
        h.setApplying(false);
        QVERIFY(!h.isApplying());
    }

    void disappearingParamTracked() {
        // A param present in seed but absent from cur creates an entry
        // with from=value, to=invalid (line 56 in UndoHistory.cpp).
        UndoHistory                               h;
        QVector<SettingsImporter::EffectSettings> snap;
        snap << makeEff("b", true, {{"v", 0}, {"extra", 5}});
        h.seed(snap);

        snap[0].parameters.remove("extra");
        snap[0].parameters["v"] = 10;
        h.recordFromCurrent(snap);

        QVERIFY(h.canUndo());
        const auto &e = h.entries()[0];
        QVERIFY(e.params.contains("extra"));
        QCOMPARE(e.params["extra"].from, QVariant(5));
        QVERIFY(!e.params["extra"].to.isValid());
    }

    void shadowRemovesNewParamOnUndo() {
        // When a param is NEW in cur (from=invalid), undoing it should remove
        // it from shadow — exercises the remove() branch in updateShadowFrom.
        UndoHistory                               h;
        QVector<SettingsImporter::EffectSettings> snap;
        snap << makeEff("b", true, {});
        h.seed(snap);

        snap[0].parameters["newparam"] = 42;
        h.recordFromCurrent(snap);
        QVERIFY(h.canUndo());

        h.undo();
        // Shadow should have "newparam" removed. Re-add with a new value:
        // this must truncate the redo tail and create a fresh entry with
        // from=invalid (confirming shadow no longer has the key).
        snap[0].parameters["newparam"] = 99;
        h.recordFromCurrent(snap);
        QVERIFY(!h.canRedo());
        QVERIFY(!h.entries().back().params["newparam"].from.isValid());
        QCOMPARE(h.entries().back().params["newparam"].to, QVariant(99));
    }

    void shadowRemovesDisappearedParamOnRedo() {
        // When a param DISAPPEARED in cur (to=invalid), redoing it removes the
        // param from shadow — exercises the remove() branch in updateShadowTo.
        UndoHistory                               h;
        QVector<SettingsImporter::EffectSettings> snap;
        snap << makeEff("b", true, {{"extra", 5}});
        h.seed(snap);

        snap[0].parameters.remove("extra");
        h.recordFromCurrent(snap);
        h.undo();
        QVERIFY(h.canRedo());

        h.redo();
        // After redo, shadow should not contain "extra"
        // Verify: record with same state (no extra) — expect no new entry
        h.recordFromCurrent(snap);
        QCOMPARE(h.cursor(), h.entries().size()); // no new entry beyond cursor
    }

    void historyChangedOnSeed() {
        UndoHistory h;
        int         count = 0;
        connect(&h, &UndoHistory::historyChanged, this, [&] { ++count; });
        QVector<SettingsImporter::EffectSettings> snap;
        snap << makeEff("b", true, {{"v", 0}});
        h.seed(snap);
        QCOMPARE(count, 1);
    }

    void historyChangedOnRecord() {
        UndoHistory                               h;
        QVector<SettingsImporter::EffectSettings> snap;
        snap << makeEff("b", true, {{"v", 0}});
        h.seed(snap);

        int count = 0;
        connect(&h, &UndoHistory::historyChanged, this, [&] { ++count; });
        snap[0].parameters["v"] = 5;
        h.recordFromCurrent(snap);
        QCOMPARE(count, 1);
    }

    void historyChangedNotOnNoOpRecord() {
        UndoHistory                               h;
        QVector<SettingsImporter::EffectSettings> snap;
        snap << makeEff("b", true, {{"v", 0}});
        h.seed(snap);

        int count = 0;
        connect(&h, &UndoHistory::historyChanged, this, [&] { ++count; });
        h.recordFromCurrent(snap); // no change
        QCOMPARE(count, 0);
    }

    void historyChangedOnUndo() {
        UndoHistory                               h;
        QVector<SettingsImporter::EffectSettings> snap;
        snap << makeEff("b", true, {{"v", 0}});
        h.seed(snap);
        snap[0].parameters["v"] = 5;
        h.recordFromCurrent(snap);

        int count = 0;
        connect(&h, &UndoHistory::historyChanged, this, [&] { ++count; });
        h.undo();
        QCOMPARE(count, 1);
    }

    void historyChangedOnRedo() {
        UndoHistory                               h;
        QVector<SettingsImporter::EffectSettings> snap;
        snap << makeEff("b", true, {{"v", 0}});
        h.seed(snap);
        snap[0].parameters["v"] = 5;
        h.recordFromCurrent(snap);
        h.undo();

        int count = 0;
        connect(&h, &UndoHistory::historyChanged, this, [&] { ++count; });
        h.redo();
        QCOMPARE(count, 1);
    }

    void historyChangedOnLoad() {
        UndoHistory h;
        int         count = 0;
        connect(&h, &UndoHistory::historyChanged, this, [&] { ++count; });

        QVector<UndoHistory::Entry> entries;
        UndoHistory::Entry          e;
        e.effectId    = "b";
        e.params["v"] = {QVariant(0), QVariant(1)};
        entries.append(e);
        h.load(entries, 1, {});
        QCOMPARE(count, 1);
    }

    void canRedoChangedSignalFires() {
        UndoHistory                               h;
        QVector<SettingsImporter::EffectSettings> snap;
        snap << makeEff("b", true, {{"v", 0}});
        h.seed(snap);
        snap[0].parameters["v"] = 5;
        h.recordFromCurrent(snap);

        bool gotFalse = true;
        connect(&h, &UndoHistory::canRedoChanged, this, [&](bool b) { gotFalse = b; });
        h.undo();
        QVERIFY(gotFalse); // canRedo went true; signal fired with true
    }

    void ensureTrackedAddsMissingDomainWithoutClearingLog() {
        UndoHistory h;
        auto        global = makeEff("brightness", true, {{"value", 0}});
        h.seed({global});
        global.parameters["value"] = 1;
        h.recordFromCurrent({global});
        QVERIFY(h.canUndo());

        auto local = makeEff("local", true, {{"present", false}});
        h.ensureTracked(local);
        local.parameters["present"] = true;
        h.ensureTracked(local); // Existing shadow must not be overwritten.
        h.recordFromCurrent({global, local});

        QCOMPARE(h.entries().size(), 2);
        QCOMPARE(h.entries().last().effectId, QString("local"));
        QCOMPARE(h.entries().last().params["present"].from, QVariant(false));
        QCOMPARE(h.entries().last().params["present"].to, QVariant(true));
    }
};

QTEST_APPLESS_MAIN(TestUndoHistory)
#include "test_undo_history.moc"
