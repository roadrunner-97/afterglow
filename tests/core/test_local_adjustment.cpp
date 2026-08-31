#include "LocalAdjustment.h"

#include <QtTest>

class TestLocalAdjustment : public QObject {
    Q_OBJECT

private slots:
    void addAssignsStableIdentityAndNames() {
        LocalAdjustmentStack stack;
        QVERIFY(stack.isEmpty());
        const QString first = stack.addLinearGradient(LinearGradientMask());
        const QString named = stack.addLinearGradient(LinearGradientMask(), "Sky");
        const QString third = stack.addLinearGradient(LinearGradientMask());
        QVERIFY(!first.isEmpty());
        QVERIFY(first != named);
        QCOMPARE(stack.adjustments().size(), 3);
        QCOMPARE(stack.find(first)->name, QString("Linear Gradient 1"));
        QCOMPARE(stack.find(named)->name, QString("Sky"));
        QCOMPARE(stack.find(third)->name, QString("Linear Gradient 2"));
    }

    void restoresOnlyValidUniqueIds() {
        LocalAdjustmentStack stack;
        LocalAdjustment      restored;
        restored.id         = "stable-id";
        restored.exposureEv = 1.25;
        restored.mask.setInverted(true);
        QVERIFY(stack.appendRestored(restored));
        QCOMPARE(stack.find("stable-id")->name, QString("Linear Gradient 1"));
        QCOMPARE(stack.find("stable-id")->exposureEv, 1.25);
        QVERIFY(stack.find("stable-id")->mask.isInverted());
        QVERIFY(!stack.appendRestored(restored));
        restored.id.clear();
        QVERIFY(!stack.appendRestored(restored));
    }

    void findConstAndMissing() const {
        LocalAdjustmentStack stack;
        const QString        id         = stack.addLinearGradient(LinearGradientMask(), "Test");
        const auto          &constStack = stack;
        QVERIFY(constStack.find(id));
        QCOMPARE(constStack.find(id)->name, QString("Test"));
        QVERIFY(!constStack.find("missing"));
        QVERIFY(!stack.find("missing"));
    }

    void removeMoveAndClear() {
        LocalAdjustmentStack stack;
        const QString        a = stack.addLinearGradient(LinearGradientMask(), "A");
        const QString        b = stack.addLinearGradient(LinearGradientMask(), "B");
        const QString        c = stack.addLinearGradient(LinearGradientMask(), "C");
        QVERIFY(!stack.move(a, -1));
        QVERIFY(!stack.move(a, 3));
        QVERIFY(!stack.move("missing", 0));
        QVERIFY(stack.move(a, 2));
        QCOMPARE(stack.adjustments()[2].id, a);
        QVERIFY(stack.move(a, 2));
        QVERIFY(stack.remove(b));
        QVERIFY(!stack.remove("missing"));
        QCOMPARE(stack.adjustments().size(), 2);
        QCOMPARE(stack.adjustments()[0].id, c);
        stack.clear();
        QVERIFY(stack.isEmpty());
    }
};

QTEST_GUILESS_MAIN(TestLocalAdjustment)
#include "test_local_adjustment.moc"
