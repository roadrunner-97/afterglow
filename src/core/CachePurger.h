#ifndef CACHEPURGER_H
#define CACHEPURGER_H

#include <QString>

namespace CachePurger {

struct Result {
    bool    success      = false;
    int     filesRemoved = 0;
    QString error;
};

// Remove only generated JPEG caches below folder. Source photos, YAML
// sidecars, history, catalog files, and application settings are untouched.
Result purgePhotoCaches(const QString &folder);

} // namespace CachePurger

#endif // CACHEPURGER_H
