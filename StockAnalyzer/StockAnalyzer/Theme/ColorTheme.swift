import SwiftUI

struct ColorTheme {
    static let background = Color(.systemBackground)
    static let secondaryBackground = Color(.secondarySystemBackground)
    static let text = Color(.label)
    static let secondaryText = Color(.secondaryLabel)
    static let accent = Color.blue
    static let success = Color.green
    static let warning = Color.orange
    static let error = Color.red
    
    static func backgroundGradient(_ opacity: Double = 1.0) -> LinearGradient {
        LinearGradient(
            gradient: Gradient(colors: [
                background.opacity(opacity),
                background.opacity(opacity * 0.8)
            ]),
            startPoint: .top,
            endPoint: .bottom
        )
    }
}
