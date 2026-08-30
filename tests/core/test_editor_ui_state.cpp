#include "EditorUiState.h"

#include <QTest>

class TestEditorUiState : public QObject {
    Q_OBJECT

private slots:
    void modeTransitionsRequireASelectionOutsideGallery() {
        EditorUiState state;
        QCOMPARE(state.mode(), EditorUiState::Mode::Gallery);
        QVERIFY(!state.requestMode(EditorUiState::Mode::Loupe, false));
        QCOMPARE(state.mode(), EditorUiState::Mode::Gallery);
        QVERIFY(state.requestMode(EditorUiState::Mode::Develop, true));
        QCOMPARE(state.mode(), EditorUiState::Mode::Develop);
        QVERIFY(state.requestMode(EditorUiState::Mode::Gallery, false));
        QCOMPARE(state.mode(), EditorUiState::Mode::Gallery);
    }

    void processingStateTracksStartAndCompletion() {
        EditorUiState state;
        QVERIFY(!state.isProcessing());
        state.setProcessing(true);
        QVERIFY(state.isProcessing());
        state.setProcessing(false);
        QVERIFY(!state.isProcessing());
    }

    void beforeViewTransitionsAreIdempotent() {
        EditorUiState state;
        QVERIFY(!state.isBeforeViewActive());
        QVERIFY(state.enterBeforeView());
        QVERIFY(state.isBeforeViewActive());
        QVERIFY(!state.enterBeforeView());
        QVERIFY(state.exitBeforeView());
        QVERIFY(!state.isBeforeViewActive());
        QVERIFY(!state.exitBeforeView());
    }

    void navigationReturnsOnlyValidNeighbours() {
        const QStringList paths{"one", "two", "three"};
        QCOMPARE(EditorUiState::navigationTarget(paths, "two", -1), QString("one"));
        QCOMPARE(EditorUiState::navigationTarget(paths, "two", 1), QString("three"));
        QVERIFY(EditorUiState::navigationTarget(paths, "one", -1).isEmpty());
        QVERIFY(EditorUiState::navigationTarget(paths, "three", 1).isEmpty());
        QVERIFY(EditorUiState::navigationTarget(paths, "missing", 1).isEmpty());
        QVERIFY(EditorUiState::navigationTarget(paths, "two", 0).isEmpty());
        QVERIFY(EditorUiState::navigationTarget({}, "two", 1).isEmpty());
        QVERIFY(EditorUiState::navigationTarget(paths, {}, 1).isEmpty());
    }
};

QTEST_APPLESS_MAIN(TestEditorUiState)
#include "test_editor_ui_state.moc"
