import SwiftUI

struct LoadingStateView: View {
    let title: String
    let message: String?
    let isLoading: Bool
    
    var body: some View {
        VStack(spacing: 16) {
            if isLoading {
                ProgressView()
                    .progressViewStyle(CircularProgressViewStyle())
                    .scaleEffect(1.5)
            }
            
            Text(title)
                .font(.headline)
                .foregroundColor(ColorTheme.text)
            
            if let message = message {
                Text(message)
                    .font(.subheadline)
                    .foregroundColor(ColorTheme.secondaryText)
                    .multilineTextAlignment(.center)
            }
        }
        .frame(maxWidth: .infinity, minHeight: 200)
        .padding()
        .background(ColorTheme.background)
    }
}
