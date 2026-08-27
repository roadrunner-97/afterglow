#include <QAction>
#include <QApplication>
#include <QSignalSpy>
#include <QStackedWidget>
#include <QTemporaryDir>
#include <QTest>

#include "EffectManager.h"
#include "GridView.h"
#include "LoupeView.h"
#include "PhotoEditorApp.h"
#include "ImageProcessor.h"

class TestGalleryLoupe : public QObject {
    Q_OBJECT

private:
    static QAction *action(PhotoEditorApp &app, const QString &text) {
        for (QAction *candidate : app.findChildren<QAction *>())
            if (candidate->text() == text) return candidate;
        return nullptr;
    }

private slots:
    void gridSelectionCanFollowLoupeNavigation() {
        GridView grid;
        grid.setPhotos({"one.jpg", "two.jpg", "three.jpg"});
        QSignalSpy changed(&grid, &GridView::currentPathChanged);

        QVERIFY(grid.setCurrentPath("three.jpg"));
        QCOMPARE(grid.currentPath(), QString("three.jpg"));
        QVERIFY(!changed.isEmpty());
        QVERIFY(!grid.setCurrentPath("missing.jpg"));
        QCOMPARE(grid.currentPath(), QString("three.jpg"));
    }

    void gridReturnsStoredThumbnail() {
        GridView grid;
        grid.setPhotos({"one.jpg"});
        QImage thumb(40, 30, QImage::Format_RGB32);
        thumb.fill(Qt::red);
        grid.setThumbnail("one.jpg", thumb);

        QVERIFY(!grid.thumbnail("one.jpg").isNull());
        QVERIFY(grid.thumbnail("missing.jpg").isNull());
    }

    void gridTracksEditedState() {
        GridView grid;
        grid.setPhotos({"one.jpg", "two.jpg"});

        QVERIFY(!grid.isEdited("one.jpg"));
        grid.setEdited("one.jpg", true);
        QVERIFY(grid.isEdited("one.jpg"));
        QVERIFY(!grid.isEdited("two.jpg"));
        grid.setEdited("one.jpg", false);
        QVERIFY(!grid.isEdited("one.jpg"));
    }

    void lateCameraDecodeDoesNotReplaceProof() {
        LoupeView loupe;
        QImage    placeholder(40, 30, QImage::Format_RGB32);
        QImage    proof(80, 60, QImage::Format_RGB32);
        QImage    decoded(160, 120, QImage::Format_RGB32);
        placeholder.fill(Qt::gray);
        proof.fill(Qt::green);
        decoded.fill(Qt::red);

        loupe.beginPhoto(placeholder);
        loupe.setProofImage(proof);
        QVERIFY(loupe.isShowingProof());
        loupe.setCameraJpegImage(decoded);
        QVERIFY(loupe.isShowingProof());
        QCOMPARE(loupe.displayedImageSize(), proof.size());
    }

    void selectingThenClickingLoupeLoadsSelectedPlaceholder() {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString firstPath  = dir.filePath("first.png");
        const QString secondPath = dir.filePath("second.png");
        QImage        first(30, 20, QImage::Format_RGB32);
        QImage        second(70, 50, QImage::Format_RGB32);
        first.fill(Qt::red);
        second.fill(Qt::blue);
        QVERIFY(first.save(firstPath));
        QVERIFY(second.save(secondPath));

        EffectManager  effects;
        PhotoEditorApp app(&effects);
        auto          *grid        = app.findChild<GridView *>();
        auto          *loupe       = app.findChild<LoupeView *>();
        auto          *stack       = app.findChild<QStackedWidget *>();
        QAction       *loupeAction = action(app, "Loupe");
        QVERIFY(grid);
        QVERIFY(loupe);
        QVERIFY(stack);
        QVERIFY(loupeAction);

        grid->setPhotos({firstPath, secondPath});
        grid->setThumbnail(firstPath, first);
        grid->setThumbnail(secondPath, second);
        QVERIFY(grid->setCurrentPath(secondPath));
        loupeAction->trigger();

        QCOMPARE(stack->currentWidget(), static_cast<QWidget *>(loupe));
        QCOMPARE(grid->currentPath(), secondPath);
        QVERIFY(!loupe->displayedImageSize().isEmpty());
        QTRY_COMPARE(loupe->displayedImageSize(), second.size());
    }

    void developWithoutSelectionDoesNotExposeOldPage() {
        EffectManager  effects;
        PhotoEditorApp app(&effects);
        auto          *grid          = app.findChild<GridView *>();
        auto          *stack         = app.findChild<QStackedWidget *>();
        QAction       *developAction = action(app, "Develop");
        QVERIFY(grid);
        QVERIFY(stack);
        QVERIFY(developAction);

        grid->setPhotos({});
        QWidget *galleryPage = stack->currentWidget();
        developAction->trigger();
        QCOMPARE(stack->currentWidget(), galleryPage);
    }

    void leavingDevelopImmediatelyUsesLatestEditedRender() {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString path = dir.filePath("photo.png");
        QImage        original(40, 30, QImage::Format_RGB32);
        original.fill(Qt::blue);
        QVERIFY(original.save(path));

        EffectManager  effects;
        PhotoEditorApp app(&effects);
        auto          *grid          = app.findChild<GridView *>();
        auto          *loupe         = app.findChild<LoupeView *>();
        auto          *processor     = app.findChild<ImageProcessor *>();
        QAction       *developAction = action(app, "Develop");
        QAction       *loupeAction   = action(app, "Loupe");
        QVERIFY(grid);
        QVERIFY(loupe);
        QVERIFY(processor);
        QVERIFY(developAction);
        QVERIFY(loupeAction);

        grid->setPhotos({path});
        grid->setThumbnail(path, original);
        QVERIFY(grid->setCurrentPath(path));
        developAction->trigger();

        QImage edited(80, 60, QImage::Format_RGB32);
        edited.fill(Qt::red);
        QVERIFY(QMetaObject::invokeMethod(processor, "processingComplete", Qt::DirectConnection, Q_ARG(QImage, edited),
                                          Q_ARG(QPoint, QPoint())));

        loupeAction->trigger();
        QVERIFY(loupe->isShowingProof());
        const QImage galleryImage = grid->thumbnail(path);
        QVERIFY(!galleryImage.isNull());
        const QColor centre = galleryImage.pixelColor(galleryImage.width() / 2, galleryImage.height() / 2);
        QVERIFY(centre.red() > centre.blue());
    }
};

QTEST_MAIN(TestGalleryLoupe)
#include "test_gallery_loupe.moc"
