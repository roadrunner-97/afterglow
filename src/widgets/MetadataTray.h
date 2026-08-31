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
        QString fileType;
        QString fileSize;
        QString dimensions;
        QString camera;
        QString lens;
        QString serial;
        QString exposure;
        QString focalLength;
        QString exposureBias;
        QString exposureProgram;
        QString meteringMode;
        QString flash;
        QString whiteBalance;
        QString colorTemperature;
        QString captured;
        QString location;
        QString creator;
        QString copyright;
        QString description;
        QString software;
    };

    explicit MetadataTray(QWidget *parent = nullptr);

    void setInfo(const Info &info);
    void clear();

private:
    QLabel *m_valFilename         = nullptr;
    QLabel *m_valFileType         = nullptr;
    QLabel *m_valFileSize         = nullptr;
    QLabel *m_valDimensions       = nullptr;
    QLabel *m_valCamera           = nullptr;
    QLabel *m_valLens             = nullptr;
    QLabel *m_valSerial           = nullptr;
    QLabel *m_valExposure         = nullptr;
    QLabel *m_valFocalLength      = nullptr;
    QLabel *m_valExposureBias     = nullptr;
    QLabel *m_valExposureProgram  = nullptr;
    QLabel *m_valMeteringMode     = nullptr;
    QLabel *m_valFlash            = nullptr;
    QLabel *m_valWhiteBalance     = nullptr;
    QLabel *m_valColorTemperature = nullptr;
    QLabel *m_valCaptured         = nullptr;
    QLabel *m_valLocation         = nullptr;
    QLabel *m_valCreator          = nullptr;
    QLabel *m_valCopyright        = nullptr;
    QLabel *m_valDescription      = nullptr;
    QLabel *m_valSoftware         = nullptr;
};

#endif // METADATATRAY_H
