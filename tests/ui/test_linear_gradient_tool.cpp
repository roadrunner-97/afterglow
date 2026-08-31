#include "LinearGradientTool.h"

#include <QImage>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QPainter>
#include <QSignalSpy>
#include <QtTest>

class TestLinearGradientTool : public QObject {
    Q_OBJECT

    static ViewportTransform transform() {
        return {{100, 100}, {100, 100}, {0.5, 0.5}, 1.0f};
    }

    static ViewportTransform portraitTransform() {
        return {{100, 200}, {100, 200}, {0.5, 0.5}, 1.0f};
    }

    static QMouseEvent mouse(QEvent::Type type, QPointF position, Qt::MouseButton button, Qt::MouseButtons buttons) {
        return {type, position, position, button, buttons, Qt::NoModifier};
    }

    static void createMask(LinearGradientTool &tool, QPointF start = {20, 50}, QPointF end = {80, 50}) {
        tool.beginCreation();
        auto press = mouse(QEvent::MouseButtonPress, start, Qt::LeftButton, Qt::LeftButton);
        QVERIFY(tool.mousePress(&press, transform()));
        auto move = mouse(QEvent::MouseMove, end, Qt::NoButton, Qt::LeftButton);
        QVERIFY(tool.mouseMove(&move, transform()));
        auto release = mouse(QEvent::MouseButtonRelease, end, Qt::LeftButton, Qt::NoButton);
        QVERIFY(tool.mouseRelease(&release, transform()));
    }

private slots:
    void portraitImageKeepsHandlesOnGradientNormal() {
        LinearGradientTool tool;
        tool.beginCreation();
        auto press = mouse(QEvent::MouseButtonPress, {20, 40}, Qt::LeftButton, Qt::LeftButton);
        QVERIFY(tool.mousePress(&press, portraitTransform()));
        auto move = mouse(QEvent::MouseMove, {80, 160}, Qt::NoButton, Qt::LeftButton);
        QVERIFY(tool.mouseMove(&move, portraitTransform()));
        auto release = mouse(QEvent::MouseButtonRelease, {80, 160}, Qt::LeftButton, Qt::NoButton);
        QVERIFY(tool.mouseRelease(&release, portraitTransform()));

        // Both endpoints remain on the dragged screen-space normal even though
        // its normalized-coordinate direction is aspect-ratio corrected.
        QCOMPARE(tool.cursorFor({20, 40}, portraitTransform()).shape(), Qt::OpenHandCursor);
        QCOMPARE(tool.cursorFor({80, 160}, portraitTransform()).shape(), Qt::OpenHandCursor);
        QVERIFY(tool.mask()->direction().y() > tool.mask()->direction().x());
    }

    void createMoveAndResize() {
        LinearGradientTool tool;
        QSignalSpy         changed(&tool, &LinearGradientTool::maskChanged);
        QSignalSpy         finished(&tool, &LinearGradientTool::gestureFinished);
        createMask(tool);
        QVERIFY(tool.hasMask());
        QVERIFY(!tool.isCreating());
        QCOMPARE(tool.mask()->center(), QPointF(0.5, 0.5));
        QCOMPARE(tool.mask()->direction(), QPointF(1.0, 0.0));
        QCOMPARE(tool.mask()->featherHalfWidth(), 0.3);
        QCOMPARE(finished.count(), 1);

        auto pressCenter = mouse(QEvent::MouseButtonPress, {50, 50}, Qt::LeftButton, Qt::LeftButton);
        QVERIFY(tool.mousePress(&pressCenter, transform()));
        auto moveCenter = mouse(QEvent::MouseMove, {60, 60}, Qt::NoButton, Qt::LeftButton);
        QVERIFY(tool.mouseMove(&moveCenter, transform()));
        auto releaseCenter = mouse(QEvent::MouseButtonRelease, {60, 60}, Qt::LeftButton, Qt::NoButton);
        QVERIFY(tool.mouseRelease(&releaseCenter, transform()));
        QCOMPARE(tool.mask()->center(), QPointF(0.6, 0.6));

        auto pressEnd = mouse(QEvent::MouseButtonPress, {90, 60}, Qt::LeftButton, Qt::LeftButton);
        QVERIFY(tool.mousePress(&pressEnd, transform()));
        auto moveEnd = mouse(QEvent::MouseMove, {100, 60}, Qt::NoButton, Qt::LeftButton);
        QVERIFY(tool.mouseMove(&moveEnd, transform()));
        auto releaseEnd = mouse(QEvent::MouseButtonRelease, {100, 60}, Qt::LeftButton, Qt::NoButton);
        QVERIFY(tool.mouseRelease(&releaseEnd, transform()));
        QCOMPARE(tool.mask()->center(), QPointF(0.65, 0.6));
        QCOMPARE(tool.mask()->featherHalfWidth(), 0.35);
        QVERIFY(changed.count() >= 4);
    }

    void cancelCreationAndMove() {
        LinearGradientTool tool;
        tool.beginCreation();
        QKeyEvent escape(QEvent::KeyPress, Qt::Key_Escape, Qt::NoModifier);
        QVERIFY(tool.keyPress(&escape));
        QVERIFY(!tool.hasMask());
        QVERIFY(!tool.isCreating());

        createMask(tool);
        const QPointF original = tool.mask()->center();
        auto          press    = mouse(QEvent::MouseButtonPress, {50, 50}, Qt::LeftButton, Qt::LeftButton);
        QVERIFY(tool.mousePress(&press, transform()));
        auto move = mouse(QEvent::MouseMove, {70, 70}, Qt::NoButton, Qt::LeftButton);
        QVERIFY(tool.mouseMove(&move, transform()));
        QVERIFY(tool.keyPress(&escape));
        QCOMPARE(tool.mask()->center(), original);
    }

    void invertVisibilityDeleteAndKeys() {
        LinearGradientTool tool;
        createMask(tool);
        tool.setInverted(true);
        QVERIFY(tool.mask()->isInverted());
        tool.setInverted(true);

        tool.setOverlayVisible(false);
        QVERIFY(!tool.isOverlayVisible());
        // Hiding the blue mask tint must not disable its on-canvas controls.
        QCOMPARE(tool.cursorFor({50, 50}, transform()).shape(), Qt::SizeAllCursor);
        tool.setOverlayVisible(false);
        tool.setOverlayVisible(true);

        QKeyEvent unrelated(QEvent::KeyPress, Qt::Key_A, Qt::NoModifier);
        QVERIFY(!tool.keyPress(&unrelated));
        QKeyEvent remove(QEvent::KeyPress, Qt::Key_Delete, Qt::NoModifier);
        QVERIFY(tool.keyPress(&remove));
        QVERIFY(!tool.hasMask());
        tool.clearMask();
        QVERIFY(!tool.keyPress(&remove));
    }

    void inputFilteringAndCursors() {
        LinearGradientTool tool;
        auto               right = mouse(QEvent::MouseButtonPress, {50, 50}, Qt::RightButton, Qt::RightButton);
        QVERIFY(!tool.mousePress(&right, transform()));
        ViewportTransform empty;
        auto              left = mouse(QEvent::MouseButtonPress, {50, 50}, Qt::LeftButton, Qt::LeftButton);
        QVERIFY(!tool.mousePress(&left, empty));
        QVERIFY(!tool.mouseMove(&left, transform()));
        QVERIFY(!tool.mouseRelease(&right, transform()));

        tool.beginCreation();
        QCOMPARE(tool.cursorFor({50, 50}, transform()).shape(), Qt::CrossCursor);
        createMask(tool);
        QCOMPARE(tool.cursorFor({50, 50}, transform()).shape(), Qt::SizeAllCursor);
        QCOMPARE(tool.cursorFor({20, 50}, transform()).shape(), Qt::OpenHandCursor);
        QCOMPARE(tool.cursorFor({5, 5}, transform()).shape(), Qt::ArrowCursor);
    }

    void paintOverlayChangesPixels() {
        LinearGradientTool tool;
        QImage             image(100, 100, QImage::Format_ARGB32_Premultiplied);
        image.fill(Qt::black);
        QPainter painter(&image);
        tool.paintOverlay(painter, transform());
        createMask(tool);
        tool.paintOverlay(painter, transform());
        painter.end();
        QVERIFY(image.pixelColor(50, 50) != QColor(Qt::black));
    }

    void overlayFeathersAndInvertMirrorsTint() {
        LinearGradientTool tool;
        createMask(tool);

        auto render = [&]() {
            QImage image(100, 100, QImage::Format_ARGB32_Premultiplied);
            image.fill(Qt::black);
            QPainter painter(&image);
            tool.paintOverlay(painter, transform());
            painter.end();
            return image;
        };

        const QImage normal = render();
        QVERIFY(normal.pixelColor(90, 25).blue() > normal.pixelColor(10, 25).blue());
        const int featherBlue = normal.pixelColor(40, 25).blue();
        QVERIFY(featherBlue > normal.pixelColor(10, 25).blue());
        QVERIFY(featherBlue < normal.pixelColor(70, 25).blue());

        tool.setInverted(true);
        const QImage inverted = render();
        QVERIFY(inverted.pixelColor(10, 25).blue() > inverted.pixelColor(90, 25).blue());
    }

    void followsViewportRotation() {
        LinearGradientTool tool;
        createMask(tool);
        ViewportTransform rotated{{100, 100}, {100, 100}, {0.5, 0.5}, 1.0f, 90.0f, {0.5, 0.5}};
        QCOMPARE(tool.cursorFor({50, 50}, rotated).shape(), Qt::SizeAllCursor);
        QCOMPARE(tool.cursorFor({50, 20}, rotated).shape(), Qt::OpenHandCursor);

        auto press = mouse(QEvent::MouseButtonPress, {50, 20}, Qt::LeftButton, Qt::LeftButton);
        QVERIFY(tool.mousePress(&press, rotated));
        auto move = mouse(QEvent::MouseMove, {50, 10}, Qt::NoButton, Qt::LeftButton);
        QVERIFY(tool.mouseMove(&move, rotated));
        auto release = mouse(QEvent::MouseButtonRelease, {50, 10}, Qt::LeftButton, Qt::NoButton);
        QVERIFY(tool.mouseRelease(&release, rotated));
        QCOMPARE(tool.mask()->center(), QPointF(0.55, 0.5));
    }
};

QTEST_MAIN(TestLinearGradientTool)
#include "test_linear_gradient_tool.moc"
