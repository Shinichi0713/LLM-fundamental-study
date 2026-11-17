#include "embedded_assets.h"

// The actual asset payload lives in the generated header.
#include "embedded_assets_data.h"
namespace llmbench {
namespace embedded {
const char* index_html() { return kIndexHtml; }
const char* styles_css() { return kStylesCss; }
const char* app_js() { return kAppJs; }
} // namespace embedded
} // namespace llmbench
