#include "LinearGradientMask.h"

#include <QtTest>
#include <cmath>

class TestLinearGradientMask : public QObject {
    Q_OBJECT

private slots:
    void defaultsAndAccessors() {
        LinearGradientMask mask;
        QCOMPARE(mask.center(), QPointF(0.5, 0.5));
        QCOMPARE(mask.direction(), QPointF(0.0, 1.0));
        QCOMPARE(mask.featherHalfWidth(), 0.25);
        QVERIFY(!mask.isInverted());

        mask.setCenter({0.2, 0.3});
        mask.setDirection({3.0, 4.0});
        mask.setFeatherHalfWidth(-0.4);
        mask.setInverted(true);
        QCOMPARE(mask.center(), QPointF(0.2, 0.3));
        QVERIFY(std::abs(mask.direction().x() - 0.6) < 1e-12);
        QVERIFY(std::abs(mask.direction().y() - 0.8) < 1e-12);
        QCOMPARE(mask.featherHalfWidth(), 0.4);
        QVERIFY(mask.isInverted());
    }

    void horizontalWeightsClampAndInterpolate() {
        LinearGradientMask mask({0.5, 0.5}, {1.0, 0.0}, 0.25);
        QCOMPARE(mask.weightAt({0.0, 0.1}), 0.0);
        QCOMPARE(mask.weightAt({0.25, 0.9}), 0.0);
        QCOMPARE(mask.weightAt({0.5, -5.0}), 0.5);
        QCOMPARE(mask.weightAt({0.625, 2.0}), 0.75);
        QCOMPARE(mask.weightAt({0.75, 0.2}), 1.0);
        QCOMPARE(mask.weightAt({2.0, 0.5}), 1.0);
    }

    void directionIsNormalized() {
        LinearGradientMask mask({0.0, 0.0}, {10.0, 0.0}, 0.5);
        QCOMPARE(mask.direction(), QPointF(1.0, 0.0));
        QCOMPARE(mask.weightAt({0.5, 0.0}), 1.0);
    }

    void diagonalAndOutOfBoundsGeometry() {
        LinearGradientMask mask({-1.0, -1.0}, {1.0, 1.0}, 2.0 * std::sqrt(2.0));
        QCOMPARE(mask.weightAt({-1.0, -1.0}), 0.5);
        QVERIFY(std::abs(mask.weightAt({0.0, 0.0}) - 0.75) < 1e-12);
        QCOMPARE(mask.weightAt({1.0, 1.0}), 1.0);
    }

    void inversionComplementsWeight() {
        LinearGradientMask normal({0.5, 0.5}, {0.0, 1.0}, 0.5, false);
        LinearGradientMask inverted({0.5, 0.5}, {0.0, 1.0}, 0.5, true);
        const QPointF      sample(0.2, 0.75);
        QCOMPARE(inverted.weightAt(sample), 1.0 - normal.weightAt(sample));
    }

    void degenerateDirectionUsesVerticalFallback() {
        LinearGradientMask mask({0.5, 0.5}, {0.0, 0.0}, 0.5);
        QCOMPARE(mask.direction(), QPointF(0.0, 1.0));
        QCOMPARE(mask.weightAt({0.5, 0.0}), 0.0);
        QCOMPARE(mask.weightAt({0.5, 1.0}), 1.0);

        mask.setDirection({0.0, 0.0});
        QCOMPARE(mask.direction(), QPointF(0.0, 1.0));
    }

    void zeroWidthIsHardTransition() {
        LinearGradientMask mask({0.5, 0.5}, {1.0, 0.0}, 0.0);
        QCOMPARE(mask.weightAt({0.49, 0.5}), 0.0);
        QCOMPARE(mask.weightAt({0.5, 0.5}), 1.0);
        QCOMPARE(mask.weightAt({0.51, 0.5}), 1.0);

        mask.setInverted(true);
        QCOMPARE(mask.weightAt({0.49, 0.5}), 1.0);
        QCOMPARE(mask.weightAt({0.5, 0.5}), 0.0);
    }
};

QTEST_GUILESS_MAIN(TestLinearGradientMask)
#include "test_linear_gradient_mask.moc"
