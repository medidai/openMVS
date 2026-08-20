// MacOpenFiles.mm
// macOS open-file bridge for GLFW app: intercepts file-open events
// from Finder / Launch Services and stores them for the main loop.
#ifdef __APPLE__
#import <Cocoa/Cocoa.h>
#import <objc/runtime.h>
#include <vector>
#include <string>

static std::vector<std::string> g_pendingFiles;

// Original finishLaunching IMP — stored during swizzle
static void (*g_origFinishLaunching)(id, SEL) = nullptr;

// IMP for application:openURLs: injected into GLFW's delegate.
// Called by macOS when files are opened via Finder / Launch Services.
static void openURLsIMP(id self, SEL _cmd, NSApplication* app, NSArray<NSURL*>* urls) {
	for (NSURL* url in urls) {
		NSString* path = [url path];
		if (path)
			g_pendingFiles.push_back([path UTF8String]);
	}
}

// Apple Event handler for kAEOpenDocuments ('odoc') — legacy fallback.
// Used when the delegate method isn't available.
@interface OpenMVSFileHandler : NSObject
@end

@implementation OpenMVSFileHandler
+ (void)handleOpenDocuments:(NSAppleEventDescriptor *)event withReplyEvent:(NSAppleEventDescriptor *)replyEvent {
	NSAppleEventDescriptor* fileList = [event paramDescriptorForKeyword:'----'];
	if (!fileList) return;
	NSInteger count = [fileList numberOfItems];
	for (NSInteger i = 1; i <= count; i++) {
		NSString* urlString = [[fileList descriptorAtIndex:i] stringValue];
		if (urlString) {
			NSURL* url = [NSURL URLWithString:urlString];
			NSString* path = url ? [url path] : urlString;
			g_pendingFiles.push_back([path UTF8String]);
		}
	}
}
@end

// Swizzled finishLaunching: injects application:openURLs: into GLFW's
// delegate BEFORE the original finishLaunching processes queued events.
// This is the key to cold-start file opening — macOS delivers the file
// event during finishLaunching, so the delegate method must exist by then.
static void patchedFinishLaunching(id self, SEL _cmd) {
	id delegate = [self delegate];
	if (delegate) {
		Class cls = [delegate class];
		if (!class_respondsToSelector(cls, @selector(application:openURLs:))) {
			class_addMethod(cls, @selector(application:openURLs:),
				(IMP)openURLsIMP, "v@:@@");
		} else {
			// GLFW (or another framework) already has the method — swizzle it
			// so we chain our handler before theirs.
			Method m = class_getInstanceMethod(cls, @selector(application:openURLs:));
			IMP origIMP = method_getImplementation(m);
			IMP newIMP = imp_implementationWithBlock(^(id _self, NSApplication* app, NSArray<NSURL*>* urls) {
				for (NSURL* url in urls) {
					NSString* path = [url path];
					if (path)
						g_pendingFiles.push_back([path UTF8String]);
				}
				((void(*)(id, SEL, NSApplication*, NSArray<NSURL*>*))origIMP)(
					_self, @selector(application:openURLs:), app, urls);
			});
			method_setImplementation(m, newIMP);
		}
	}
	// Call the original finishLaunching
	if (g_origFinishLaunching)
		g_origFinishLaunching(self, _cmd);
}

extern "C" void OpenMVS_InstallFileHandler() {
	@autoreleasepool {
		// 1) Register legacy Apple Event handler for 'odoc' as a fallback
		[[NSAppleEventManager sharedAppleEventManager]
			setEventHandler:[OpenMVSFileHandler class]
			andSelector:@selector(handleOpenDocuments:withReplyEvent:)
			forEventClass:'aevt'
			andEventID:'odoc'];

		// 2) Swizzle NSApplication's finishLaunching so that when glfwInit()
		//    calls it, we inject application:openURLs: into the delegate
		//    BEFORE macOS processes queued file-open events.
		Method method = class_getInstanceMethod([NSApplication class],
			@selector(finishLaunching));
		g_origFinishLaunching = (void(*)(id, SEL))method_getImplementation(method);
		method_setImplementation(method, (IMP)patchedFinishLaunching);
	}
}

extern "C" void OpenMVS_ConsumePendingOpenFiles(std::vector<std::string>& out) {
	out.swap(g_pendingFiles);
	g_pendingFiles.clear();
}
#endif
