#ifndef METADATATRAY_H
#define METADATATRAY_H

#include <QString>
#include <QWidget>

class QLabel;

class MetadataTray : public QWidget {
    Q_OBJECT
public:
    struct Info {
        QString filename;
        QString dimensions;
        QString camera;
        QString lens;
        QString exposure;
        QString captured;
    };

    explicit MetadataTray(QWidget *parent = nullptr);

    void setInfo(const Info &info);
    void clear();

private:
    QLabel *m_valFilename   = nullptr;
    QLabel *m_valDimensions = nullptr;
    QLabel *m_valCamera     = nullptr;
    QLabel *m_valLens       = nullptr;
    QLabel *m_valExposure   = nullptr;
    QLabel *m_valCaptured   = nullptr;
};

#endif // METADATATRAY_H
