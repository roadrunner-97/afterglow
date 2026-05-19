#include <QApplication>
#include <QListWidget>
#include <QTest>
#include "HistoryTray.h"

static QVector<HistoryTray::Row> makeRows(const QStringList& labels)
{
    QVector<HistoryTray::Row> rows;
    for (const auto& l : labels)
        rows.append({l});
    return rows;
}

// Find the QListWidget inside the tray
static QListWidget* listOf(HistoryTray* t)
{
    return t->findChild<QListWidget*>();
}

class TestHistoryTray : public QObject {
    Q_OBJECT

private slots:
    void emptyHistoryHasOriginalOnly() {
        HistoryTray tray;
        tray.setHistory({}, 0);
        QListWidget* list = listOf(&tray);
        QVERIFY(list);
        QCOMPARE(list->count(), 1);
        QCOMPARE(list->item(0)->text(), QString("Original"));
    }

    void rowCountIncludesOriginal() {
        HistoryTray tray;
        tray.setHistory(makeRows({"Brightness value 0 → 10",
                                  "Vignette on",
                                  "Saturation value 0 → 5"}), 3);
        QListWidget* list = listOf(&tray);
        QCOMPARE(list->count(), 4);   // Original + 3 entries
    }

    void currentRowIsBoldAndSelected() {
        HistoryTray tray;
        tray.setHistory(makeRows({"A", "B"}), 2);   // cursor at last entry
        QListWidget* list = listOf(&tray);
        // List index 2 = cursor 2 = entry "B"
        QListWidgetItem* current = list->item(2);
        QVERIFY(current);
        QVERIFY(current->font().bold());
        QCOMPARE(list->currentRow(), 2);
    }

    void originalIsCurrentWhenCursorZero() {
        HistoryTray tray;
        tray.setHistory(makeRows({"A", "B"}), 0);
        QListWidget* list = listOf(&tray);
        QCOMPARE(list->currentRow(), 0);
        QVERIFY(list->item(0)->font().bold());
    }

    void redoTailIsDimmed() {
        HistoryTray tray;
        // cursor=1: Original (0) and "A" (1) are applied; "B" (2) is redo tail
        tray.setHistory(makeRows({"A", "B"}), 1);
        QListWidget* list = listOf(&tray);
        // Item 2 ("B") should have a different (dimmed) foreground
        const QColor normal  = list->item(0)->foreground().color();
        const QColor dimmed  = list->item(2)->foreground().color();
        // dimmed must differ from the default foreground; it's the disabled text colour
        QVERIFY(normal != dimmed || dimmed.alpha() == 0);
        // items 0 and 1 are NOT in the redo tail — their foreground must not be dimmed
        QCOMPARE(list->item(1)->foreground().color(), normal);
    }

    void rowActivatedIndexMapping() {
        HistoryTray tray;
        tray.show();   // needed for itemClicked to work in tests
        tray.setHistory(makeRows({"A", "B", "C"}), 3);
        QListWidget* list = listOf(&tray);

        int lastActivated = -1;
        connect(&tray, &HistoryTray::rowActivated,
                this, [&](int i){ lastActivated = i; });

        // Simulate click on "Original" (index 0)
        emit list->itemClicked(list->item(0));
        QCOMPARE(lastActivated, 0);

        // Simulate click on third entry (index 3)
        emit list->itemClicked(list->item(3));
        QCOMPARE(lastActivated, 3);
    }

    void rebuildPreservesCorrectCount() {
        HistoryTray tray;
        tray.setHistory(makeRows({"A"}), 1);
        tray.setHistory(makeRows({"A", "B", "C"}), 2);
        QListWidget* list = listOf(&tray);
        QCOMPARE(list->count(), 4);
        QCOMPARE(list->currentRow(), 2);
    }
};

QTEST_MAIN(TestHistoryTray)
#include "test_history_tray.moc"
