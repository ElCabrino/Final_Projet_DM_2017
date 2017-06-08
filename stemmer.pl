#!/usr/bin/perl

# Description: Perform stemming of words in STDIN into STDOUT.
# It uses the Snowball perl library.
# Author: Adam Woznica
# Date: April 2012
# Usage: ./stemmer.pl <-d>
# -d: disable the stemming

use strict;
use warnings;
use Lingua::Stem::Snowball;
use Getopt::Long;

my $disable = 0;

sub showusage {
    print STDERR <<USAGE
Usage: $0 <-d>.

    -d: disable the stemming
USAGE
      ;
    exit(1);
}
GetOptions( "d" => \$disable );

foreach (<STDIN>) {
	if ($disable) {
		print $_;
	} else {
		tr/[A-Z]/[a-z]/;
		my @words = split('\s+');
		my $stemmer = Lingua::Stem::Snowball->new( lang => 'en' );
    		$stemmer->stem_in_place( \@words ); 
		print join(' ', @words) . "\n";
	}
}

