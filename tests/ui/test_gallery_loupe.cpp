#include <QAction>
#include <QApplication>
#include <QSignalSpy>
#include <QStackedWidget>
#include <QListWidget>
#include <QTemporaryDir>
#include <QTest>
#include <QWheelEvent>
#include <QPushButton>

#include "EffectManager.h"
#include "GridView.h"
#include "LoupeView.h"
#include "PhotoEditorApp.h"
#include "ImageProcessor.h"
#include "UiServices.h"

class FakeUiServices final : public UiServices {
public:
    QString                               openFileResult;
    QString                               saveFileResult;
    QString                               directoryResult;
    std::optional<ExportOptions::Options> exportOptions;
    bool                                  confirmResult = false;
    QStringList                           informationTitles;
    QStringList                           warningTitles;

    QString openFile(QWidget *, const QString &, const QString &, const QString &) override {
        return openFileResult;
    }
    QString saveFile(QWidget *, const QString &, const QString &, const QString &) override {
        return saveFileResult;
    }
    QString chooseDirectory(QWidget *, const QString &, const QString &) override {
        return directoryResult;
    }
    std::optional<ExportOptions::Options> chooseExportOptions(QWidget *, const QString &) override {
        return exportOptions;
    }
    void information(QWidget *, const QString &title, const QString &) override {
        informationTitles.append(title);
    }
    void warning(QWidget *, const QString &title, const QString &) override {
        warningTitles.append(title);
    }
    bool confirm(QWidget *, const QString &, const QString &) override {
        return confirmResult;
    }
};

class TestGalleryLoupe : public QObject {
    Q_OBJECT

private:
    static QAction *action(PhotoEditorApp &app, const QString &text) {
        for (QAction *candidate : app.findChildren<QAction *>())
            if (candidate->text() == text) return candidate;
        return nullptr;
    }

private slots:
    void importantControlsHaveStableObjectNames() {
        EffectManager  effects;
        PhotoEditorApp app(&effects);

        QVERIFY(app.findChild<QAction *>("actionOpenImage"));
        QVERIFY(app.findChild<QAction *>("actionOpenFolder"));
        QVERIFY(app.findChild<QAction *>("actionSaveImage"));
        QVERIFY(app.findChild<QAction *>("actionUndo"));
        QVERIFY(app.findChild<QAction *>("actionRedo"));
        QVERIFY(app.findChild<QAction *>("actionModeGallery"));
        QVERIFY(app.findChild<QAction *>("actionModeLoupe"));
        QVERIFY(app.findChild<QAction *>("actionModeDevelop"));
        QVERIFY(app.findChild<QStackedWidget *>("editorModeStack"));
        QVERIFY(app.findChild<GridView *>("galleryGrid"));
        QVERIFY(app.findChild<LoupeView *>("loupeView"));
        QVERIFY(app.findChild<ViewportWidget *>("developViewport"));
        QVERIFY(app.findChild<QWidget *>("processingIndicator"));
        QVERIFY(app.findChild<QWidget *>("gpuDeviceSelector"));
    }

    void injectedOpenImageRunsDevelopWorkflowWithoutNativeDialogs() {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString imagePath = dir.filePath("workflow.png");
        QImage        image(48, 32, QImage::Format_RGB32);
        image.fill(Qt::cyan);
        QVERIFY(image.save(imagePath));

        FakeUiServices ui;
        EffectManager  effects;
        PhotoEditorApp app(&effects);
        ui.openFileResult = imagePath;
        app.setUiServices(&ui);

        app.findChild<QAction *>("actionOpenImage")->trigger();

        auto *stack      = app.findChild<QStackedWidget *>("editorModeStack");
        auto *viewport   = app.findChild<ViewportWidget *>("developViewport");
        auto *processing = app.findChild<QWidget *>("processingIndicator");
        QVERIFY(stack);
        QVERIFY(viewport);
        QVERIFY(processing);
        QCOMPARE(stack->currentIndex(), static_cast<int>(EditorUiState::Mode::Develop));
        QVERIFY(stack->currentWidget()->isAncestorOf(viewport));
        QTRY_VERIFY(!processing->isVisible());
        QVERIFY(QFile::exists(dir.filePath("workflow.yml")));
    }

    void explicitImagePathRunsDevelopWorkflowWithoutNativeDialogs() {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString imagePath = dir.filePath("command-line.png");
        QImage        image(48, 32, QImage::Format_RGB32);
        image.fill(Qt::cyan);
        QVERIFY(image.save(imagePath));

        EffectManager  effects;
        PhotoEditorApp app(&effects);
        app.openImagePath(imagePath);

        auto *stack      = app.findChild<QStackedWidget *>("editorModeStack");
        auto *processing = app.findChild<QWidget *>("processingIndicator");
        QVERIFY(stack);
        QVERIFY(processing);
        QCOMPARE(stack->currentIndex(), static_cast<int>(EditorUiState::Mode::Develop));
        QTRY_VERIFY(!processing->isVisible());
        QVERIFY(QFile::exists(dir.filePath("command-line.yml")));
    }

    void injectedServicesDriveSettingsAndExportWorkflows() {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString imagePath = dir.filePath("source.png");
        QImage        image(40, 30, QImage::Format_RGB32);
        image.fill(Qt::magenta);
        QVERIFY(image.save(imagePath));

        FakeUiServices ui;
        EffectManager  effects;
        PhotoEditorApp app(&effects);
        app.setUiServices(&ui);

        ui.openFileResult = imagePath;
        app.findChild<QAction *>("actionOpenImage")->trigger();

        const QString settingsPath = dir.filePath("saved-settings.yml");
        ui.saveFileResult          = settingsPath;
        app.findChild<QAction *>("actionSaveSettings")->trigger();
        QVERIFY(QFile::exists(settingsPath));

        ui.openFileResult = dir.filePath("missing-settings.yml");
        app.findChild<QAction *>("actionLoadSettings")->trigger();
        QCOMPARE(ui.warningTitles, QStringList{"Load Failed"});

        ExportOptions::Options options;
        options.destinationDir  = dir.path();
        options.filenamePattern = "rendered";
        options.format          = ExportOptions::Format::PNG;
        options.onConflict      = ExportOptions::OverwritePolicy::Overwrite;
        ui.exportOptions        = options;
        app.findChild<QAction *>("actionSaveImage")->trigger();
        // A cold POCL runner may spend several seconds compiling the shared
        // pipeline while the rest of ctest is running in parallel.
        QTRY_VERIFY_WITH_TIMEOUT(QFile::exists(dir.filePath("rendered.png")), 15000);
        QCOMPARE(ui.warningTitles, QStringList{"Load Failed"});
    }

    void injectedFolderPickerRunsGalleryWorkflow() {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString imagePath = dir.filePath("gallery.png");
        QImage        image(32, 24, QImage::Format_RGB32);
        image.fill(Qt::yellow);
        QVERIFY(image.save(imagePath));

        FakeUiServices ui;
        EffectManager  effects;
        PhotoEditorApp app(&effects);
        ui.directoryResult = dir.path();
        app.setUiServices(&ui);

        app.findChild<QAction *>("actionOpenFolder")->trigger();

        auto *stack = app.findChild<QStackedWidget *>("editorModeStack");
        auto *grid  = app.findChild<GridView *>("galleryGrid");
        QVERIFY(stack);
        QVERIFY(grid);
        QCOMPARE(stack->currentIndex(), static_cast<int>(EditorUiState::Mode::Gallery));
        QCOMPARE(grid->currentPath(), imagePath);
        QTRY_VERIFY(!grid->thumbnail(imagePath).isNull());
    }

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

    void gridCapsStoredThumbnailsAtMaximumZoomSize() {
        GridView grid;
        grid.setPhotos({"large.jpg"});
        QImage large(1600, 1200, QImage::Format_RGB32);
        large.fill(Qt::blue);

        grid.setThumbnail("large.jpg", large);

        const QImage stored = grid.thumbnail("large.jpg");
        QVERIFY(!stored.isNull());
        QVERIFY(stored.width() <= 512);
        QVERIFY(stored.height() <= 512);
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

    void galleryMarkToggleMatchesLoupeSemantics() {
        GridView   grid;
        QSignalSpy changed(&grid, &GridView::markChanged);
        grid.setPhotos({"one.jpg"});

        grid.toggleMark("one.jpg", GridView::Mark::Accept);
        QCOMPARE(grid.mark("one.jpg"), GridView::Mark::Accept);
        grid.toggleMark("one.jpg", GridView::Mark::Accept);
        QCOMPARE(grid.mark("one.jpg"), GridView::Mark::None);
        grid.toggleMark("one.jpg", GridView::Mark::Decline);
        QCOMPARE(grid.mark("one.jpg"), GridView::Mark::Decline);
        QCOMPARE(changed.count(), 3);
    }

    void controlWheelOnGalleryChangesThumbnailSize() {
        GridView grid;
        grid.setPhotos({"one.jpg"});
        auto *list = grid.findChild<QListWidget *>();
        QVERIFY(list);
        const QSize initial = list->iconSize();

        QWheelEvent zoomIn(QPointF(10, 10), QPointF(10, 10), QPoint(), QPoint(0, 120), Qt::NoButton,
                           Qt::ControlModifier, Qt::NoScrollPhase, false);
        QApplication::sendEvent(list->viewport(), &zoomIn);
        QCOMPARE(list->iconSize(), initial + QSize(16, 16));

        QWheelEvent ordinaryScroll(QPointF(10, 10), QPointF(10, 10), QPoint(), QPoint(0, -120), Qt::NoButton,
                                   Qt::NoModifier, Qt::NoScrollPhase, false);
        QApplication::sendEvent(list->viewport(), &ordinaryScroll);
        QCOMPARE(list->iconSize(), initial + QSize(16, 16));
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

    void imageVersionButtonsAreMutuallyExclusive() {
        LoupeView loupe;
        QImage    camera(40, 30, QImage::Format_RGB32);
        QImage    original(80, 60, QImage::Format_RGB32);
        QImage    edited(120, 90, QImage::Format_RGB32);
        loupe.beginPhoto(camera);
        loupe.setOriginalRawImage(original);
        loupe.setProofImage(edited);

        auto button = [&loupe](const QString &text) {
            for (auto *candidate : loupe.findChildren<QPushButton *>())
                if (candidate->text() == text) return candidate;
            return static_cast<QPushButton *>(nullptr);
        };
        auto *cameraButton   = button("Camera JPEG");
        auto *originalButton = button("Original RAW");
        auto *editedButton   = button("Edited RAW");
        QVERIFY(cameraButton);
        QVERIFY(originalButton);
        QVERIFY(editedButton);
        QVERIFY(editedButton->isChecked());

        originalButton->click();
        QCOMPARE(loupe.displayedImageSize(), original.size());
        QVERIFY(originalButton->isChecked());
        QVERIFY(!editedButton->isChecked());

        cameraButton->click();
        QCOMPARE(loupe.displayedImageSize(), camera.size());
        QVERIFY(cameraButton->isChecked());
        QVERIFY(!originalButton->isChecked());
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

    void debugMenuExposesEditedPreviewRebuild() {
        EffectManager  effects;
        PhotoEditorApp app(&effects);
        QAction       *rebuildAction = action(app, "Rebuild Edited Previews");
        QVERIFY(rebuildAction);
    }

    void developSettingsActionsExposeClipboardShortcuts() {
        EffectManager  effects;
        PhotoEditorApp app(&effects);
        QAction       *copyAction  = action(app, "Copy Develop Settings");
        QAction       *pasteAction = action(app, "Paste Develop Settings");
        QVERIFY(copyAction);
        QVERIFY(pasteAction);
        QCOMPARE(copyAction->shortcut(), QKeySequence::Copy);
        QCOMPARE(pasteAction->shortcut(), QKeySequence::Paste);
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
