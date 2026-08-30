#ifndef EDITORUISTATE_H
#define EDITORUISTATE_H

#include <QString>
#include <QStringList>

// Headless state and transition rules shared by PhotoEditorApp's controls.
// Keeping these decisions out of QWidget code makes navigation and transient
// UI state deterministic to test without a display or event loop.
class EditorUiState {
public:
    enum class Mode { Gallery = 0, Loupe = 1, Develop = 2 };

    Mode mode() const;
    bool requestMode(Mode requested, bool hasSelection);

    bool isProcessing() const;
    void setProcessing(bool processing);

    bool isBeforeViewActive() const;
    bool enterBeforeView();
    bool exitBeforeView();

    static QString navigationTarget(const QStringList &paths, const QString &currentPath, int direction);

private:
    Mode m_mode             = Mode::Gallery;
    bool m_processing       = false;
    bool m_beforeViewActive = false;
};

#endif // EDITORUISTATE_H
