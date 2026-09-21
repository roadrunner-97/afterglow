#ifndef LINEARIMAGEIO_H
#define LINEARIMAGEIO_H

#include <QImage>
#include <QString>

namespace LinearImageIO {

bool   isExrPath(const QString &path);
QImage readExr(const QString &path, QString *error = nullptr);
bool   writeExr(const QString &path, const QImage &image, QString *error = nullptr);

} // namespace LinearImageIO

#endif // LINEARIMAGEIO_H
