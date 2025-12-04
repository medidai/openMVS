// MacOpenFiles.mm
// Minimal macOS open-file bridge for GLFW app
#ifdef __APPLE__
#import <Cocoa/Cocoa.h>
#import <objc/runtime.h>
#include <vector>
#include <string>

static std::vector<std::string> g_pendingFiles;

// Helper class to handle Apple Events
@interface OpenMVSFileHandler : NSObject
@end

@implementation OpenMVSFileHandler
+ (void)handleOpenDocuments:(NSAppleEventDescriptor *)event withReplyEvent:(NSAppleEventDescriptor *)replyEvent {
    // '----' is keyDirectObject
    NSAppleEventDescriptor* fileList = [event paramDescriptorForKeyword:'----'];
    if (!fileList) return;
    
    NSInteger count = [fileList numberOfItems];
    for (NSInteger i = 1; i <= count; i++) {
        NSString* urlString = [[fileList descriptorAtIndex:i] stringValue];
        if (urlString) {
            // Convert file:// URL to filesystem path
            NSURL* url = [NSURL URLWithString:urlString];
            NSString* path = url ? [url path] : urlString;
            g_pendingFiles.push_back([path UTF8String]);
        }
    }
}
@end

extern "C" void OpenMVS_InstallFileHandler() {
    @autoreleasepool {
        // Register the Apple Event handler for kAEOpenDocuments ('odoc')
        // This bypasses the NSApplicationDelegate and works even before NSApp is fully initialized
        [[NSAppleEventManager sharedAppleEventManager] 
            setEventHandler:[OpenMVSFileHandler class] 
            andSelector:@selector(handleOpenDocuments:withReplyEvent:) 
            forEventClass:'aevt' // kCoreEventClass
            andEventID:'odoc'];  // kAEOpenDocuments
    }
}

extern "C" void OpenMVS_ConsumePendingOpenFiles(std::vector<std::string>& out) {
    out.swap(g_pendingFiles);
    g_pendingFiles.clear();
}
#endif
