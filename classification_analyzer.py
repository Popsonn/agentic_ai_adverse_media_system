#!/usr/bin/env python3
"""
Classification Results Analyzer
Run this script to view detailed classification results from your adverse media system
"""

import sys
import json
from datetime import datetime
from main import create_system  # Your actual system

def display_classification_results(result):
    """Display detailed classification results from your actual system output"""
    
    print("\n" + "="*80)
    print("🔍 ADVERSE MEDIA CLASSIFICATION ANALYSIS")
    print("="*80)
    
    # Basic info from your actual result structure
    entity_name = result.get('entity_name', 'Unknown')
    status = result.get('status', 'Unknown')
    
    print(f"📋 Entity: {entity_name}")
    print(f"✅ Status: {status}")
    
    # Get articles from your actual structure: result['classifications'] contains resolved_articles
    articles = result.get('classifications', [])
    
    if not articles:
        print("\n❌ No classified articles found!")
        print("\nAvailable result keys:")
        for key in sorted(result.keys()):
            print(f"  - {key}")
        return
    
    print(f"\n📊 Total Articles Found: {len(articles)}")
    
    # Count by categories - using your actual nested structure
    adverse_count = 0
    category_counts = {}
    involvement_counts = {}
    
    for article in articles:
        # Your structure: top-level is_deemed_adverse OR nested in classified_article
        is_adverse = article.get('is_deemed_adverse', False)
        if not is_adverse:
            classified_article = article.get('classified_article', {})
            is_adverse = classified_article.get('is_deemed_adverse', False)
        
        if is_adverse:
            adverse_count += 1
        
        # Get category and involvement from your actual structure
        category = article.get('final_adverse_category', 'UNKNOWN')
        involvement = article.get('final_entity_involvement', 'UNKNOWN')
        
        category_counts[category] = category_counts.get(category, 0) + 1
        involvement_counts[involvement] = involvement_counts.get(involvement, 0) + 1
    
    print(f"🚨 Adverse Articles: {adverse_count}")
    print(f"✅ Clean Articles: {len(articles) - adverse_count}")
    
    # Category breakdown
    if category_counts:
        print(f"\n📂 Categories Found:")
        for category, count in sorted(category_counts.items()):
            print(f"  • {category}: {count} articles")
    
    # Involvement level breakdown
    if involvement_counts:
        print(f"\n👤 Entity Involvement Levels:")
        for involvement, count in sorted(involvement_counts.items()):
            print(f"  • {involvement}: {count} articles")
    
    # Detailed article breakdown using your actual structure
    print(f"\n📄 DETAILED ARTICLE ANALYSIS")
    print("="*80)
    
    for i, article in enumerate(articles, 1):
        # Extract from your nested structure
        classified_article = article.get('classified_article', {})
        
        # Try different title fields from your structure
        title = (classified_article.get('article_title') or 
                classified_article.get('title') or 
                'Unknown Title')
        
        url = classified_article.get('url', 'No URL')
        source = classified_article.get('source', 'Unknown Source')
        
        # Get classification results from your structure
        category = article.get('final_adverse_category', 'UNKNOWN')
        involvement = article.get('final_entity_involvement', 'UNKNOWN')
        confidence = article.get('final_overall_confidence', 0)
        
        # Check adverse status at both levels
        is_adverse = article.get('is_deemed_adverse', False)
        if not is_adverse:
            is_adverse = classified_article.get('is_deemed_adverse', False)
        
        status_emoji = "🚨" if is_adverse else "✅"
        
        print(f"\n{status_emoji} Article {i}: {title[:70]}{'...' if len(title) > 70 else ''}")
        print(f"   Category: {category}")
        print(f"   Involvement: {involvement}")
        print(f"   Confidence: {confidence:.2f}")
        print(f"   Source: {source}")
        print(f"   URL: {url}")
        
        # Show resolution details from your structure
        resolution_method = article.get('resolution_method')
        if resolution_method:
            print(f"   Resolution: {resolution_method}")
        
        # Show any review requirements
        if article.get('requires_further_human_review', False):
            review_reason = article.get('further_review_reason', 'No reason specified')
            print(f"   ⚠️  Requires Review: {review_reason}")

def export_to_json(result, filename=None):
    """Export results to JSON file"""
    if not filename:
        entity_name = result.get('entity_name', 'unknown').replace(' ', '_')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"classification_results_{entity_name}_{timestamp}.json"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n💾 Results exported to: {filename}")
        return filename
    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        return None

def main():
    """Main function using your actual system"""
    
    if len(sys.argv) < 2:
        print("Usage: python classification_analyzer.py \"Entity Name\" [--context \"additional context\"] [--export]")
        print("Example: python classification_analyzer.py \"Do Kwon\" --context \"CEO of Terraform Labs\"")
        sys.exit(1)
    
    entity_name = sys.argv[1]
    context = None
    export_results = False
    
    # Parse arguments
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--context' and i + 1 < len(sys.argv):
            context = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--export':
            export_results = True
            i += 1
        else:
            i += 1
    
    print(f"🔍 Analyzing adverse media for: {entity_name}")
    if context:
        print(f"📝 Context: {context}")
    
    try:
        # Use your actual system
        print("\n⚙️  Initializing adverse media system...")
        system = create_system()
        
        print("🚀 Running analysis...")
        # Use your actual method with context support
        result = system.process_entity_with_context_support(entity_name, context)
        
        # Display results using your actual structure
        display_classification_results(result)
        
        # Export if requested
        if export_results:
            export_to_json(result)
        
        # Summary using your actual structure
        articles = result.get('classifications', [])
        adverse_count = 0
        for article in articles:
            is_adverse = article.get('is_deemed_adverse', False)
            if not is_adverse:
                classified_article = article.get('classified_article', {})
                is_adverse = classified_article.get('is_deemed_adverse', False)
            if is_adverse:
                adverse_count += 1
        
        print(f"\n🎯 SUMMARY: Found {len(articles)} articles, {adverse_count} adverse")
        
        # Handle context requests from your system
        if result.get('status') == 'NEEDS_CONTEXT':
            print(f"\n📝 CONTEXT REQUIRED:")
            context_req = result.get('context_request', {})
            print(f"Message: {context_req.get('message', 'Additional context needed')}")
            suggested = context_req.get('suggested_fields', [])
            if suggested:
                print(f"Suggested fields: {', '.join(suggested)}")
            print(f"\nTo provide context, run:")
            print(f"python main.py '{entity_name}' --dob YYYY-MM-DD --occupation 'Job Title'")
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()