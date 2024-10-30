import SwiftUI

struct SlideTransition: ViewModifier {
    let isPresented: Bool
    
    func body(content: Content) -> some View {
        content
            .opacity(isPresented ? 1 : 0)
            .offset(y: isPresented ? 0 : 20)
            .animation(.easeInOut(duration: 0.3), value: isPresented)
    }
}

extension View {
    func slideTransition(isPresented: Bool) -> some View {
        modifier(SlideTransition(isPresented: isPresented))
    }
}
