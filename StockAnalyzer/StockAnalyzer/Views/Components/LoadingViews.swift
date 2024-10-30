import SwiftUI

struct LoadingView: View {
    var body: some View {
        VStack(spacing: 20) {
            ProgressView()
                .progressViewStyle(CircularProgressViewStyle())
                .scaleEffect(1.5)
            
            Text("Analyzing...")
                .font(.headline)
                .foregroundColor(ColorTheme.text)
            
            Text("Please wait while we process the data")
                .font(.subheadline)
                .foregroundColor(ColorTheme.secondaryText)
                .multilineTextAlignment(.center)
        }
        .padding()
        .frame(maxWidth: .infinity, minHeight: 200)
        .background(ColorTheme.background)
        .cornerRadius(10)
        .shadow(radius: 2)
    }
}

#Preview {
    LoadingView()
}
