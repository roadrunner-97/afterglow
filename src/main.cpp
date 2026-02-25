#include <QApplication>
#include "ui/PhotoEditorApp.h"

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    PhotoEditorApp window;
    window.show();

    return app.exec();
}
